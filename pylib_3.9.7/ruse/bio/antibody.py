import collections
from dataclasses import dataclass
from enum import Enum  # consider enum.StrEnum for Python >= 3.11
import json
import os
import re
import traceback
import uuid
from typing import List, NamedTuple, Optional, Dict, Final, Tuple

import numpy as np

from Bio.PDB.Structure import Structure
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from df.data_transfer import Notification, NotificationLevel

from df.df_util import euclidean_distance_matrix

from ruse.bio.applications import AnarciCommandLine
from ruse.bio.bio_data_table_helper import sequence_to_genbank_base64_str
from ruse.bio.bio_util import sequences_to_file
from ruse.bio.sequence_align import copy_features_to_gapped_sequence

ANTIBODY_NUMBERING_COLUMN_PROPERTY: Final[str] = "antibodyNumbering"

class AntibodyNumber(NamedTuple):
    domain: int
    chain: str
    position: int
    insertion: Optional[str]
    query_position: Optional[int]

    def label(self):
        insertion = self.insertion if self.insertion else ''
        return '{}{}{}'.format(self.chain, self.position, insertion)

    def __lt__(self, other: 'AntibodyNumber') -> bool:
        if self.domain != other.domain:
            return self.domain < other.domain
        if self.chain != other.chain:
            return self.chain == 'L'
        if self.position != other.position:
            return self.position < other.position
        if self.insertion is None and other.insertion:
            return True
        if self.insertion and other.insertion:
            return self.insertion < other.insertion
        return False

    def __eq__(self, other: 'AntibodyNumber') -> bool:
        return self.domain == other.domain and self.chain == other.chain and self.position == other.position \
               and self.insertion == other.insertion


class AntibodyNumberMapping(NamedTuple):
    domain: int
    chain: str
    position: int
    insertion: Optional[str]
    residue: str
    query_position: Optional[int]

    def label(self):
        insertion = self.insertion if self.insertion else ''
        return '{}{}{}'.format(self.chain, self.position, insertion)

    def to_antibody_number(self) -> AntibodyNumber:
        return AntibodyNumber(self.domain, self.chain, self.position, self.insertion, self.query_position)

    def matches(self, number: AntibodyNumber) -> bool:
        return self.chain == number.chain and self.position == number.position and self.insertion and number.insertion

class NumberingScheme(Enum):
    """
        An enum to specify the antibody numbering scheme

        Members:
            KABAT
            CHOTHIA
            IMGT
            MARTIN
            AHO
    """

    KABAT = 'kabat'
    CHOTHIA  = 'chothia'
    IMGT = 'imgt'
    MARTIN = 'martin'
    AHO = 'aho'

    @classmethod
    def from_str(cls, scheme: str):
        if scheme.upper() not in cls.__members__:
            raise NotImplementedError

        return cls[scheme.upper()]

    def __str__(self):
        return self.value

class CDRDefinitionScheme(Enum):
    """
        An enum to specify the antibody CDR definition scheme

        Members:
            KABAT
            CHOTHIA
            IMGT
            IMGT_LEFRANC - added 24.03.11 to correspond with
                        * Lefranc, Immunology Today 18, 509 (1997)
                        * Lefranc, The Immunologist 7, 132-136 (1999)
                        * Lefranc, et al. Dev. Comp. Immun. 27, 55-77 (2003)
    """

    KABAT = 'kabat'
    CHOTHIA  = 'chothia'
    IMGT = 'imgt'
    IMGT_LEFRANC = 'imgt_lefranc'

    @classmethod
    def from_str(cls, scheme: str):
        if scheme.upper() not in cls.__members__:
            raise NotImplementedError

        return cls[scheme.upper()]

    def __str__(self):
        return self.value


class ChainRegions(NamedTuple):
    domain: int
    chain: str
    start: AntibodyNumber
    end: AntibodyNumber
    cdr1_start: AntibodyNumber
    cdr1_end: AntibodyNumber
    cdr2_start: AntibodyNumber
    cdr2_end: AntibodyNumber
    cdr3_start: AntibodyNumber
    cdr3_end: AntibodyNumber

    def to_data(self) -> Dict:
        start = self.start.query_position
        end = self.end.query_position
        start1 = self.cdr1_start.query_position
        end1 = self.cdr1_end.query_position
        start2 = self.cdr2_start.query_position
        end2 = self.cdr2_end.query_position
        start3 = self.cdr3_start.query_position
        end3 = self.cdr3_end.query_position
        chain = self.chain
        data = list()

        data.append({'name': '{}FR1'.format(chain), 'start': start, 'end': start1})
        data.append({'name': 'CDR-{}1'.format(chain), 'start': start1, 'end': end1 + 1})
        data.append({'name': '{}FR2'.format(chain), 'start': end1 + 1, 'end': start2})
        data.append({'name': 'CDR-{}2'.format(chain), 'start': start2, 'end': end2 + 1})
        data.append({'name': '{}FR3'.format(chain), 'start': end2 + 1, 'end': start3})
        data.append({'name': 'CDR-{}3'.format(chain), 'start': start3, 'end': end3 + 1})
        data.append({'name': '{}FR4'.format(chain), 'start': end3 + 1, 'end': end + 1})

        return {'domain': self.domain, 'chain': self.chain, 'regions': data}


class AntibodyAlignmentResult(NamedTuple):
    aligned_sequences: Optional[List[SeqRecord]]
    numbering: List[AntibodyNumber]
    regions: List[ChainRegions]
    numbering_scheme: NumberingScheme
    cdr_definition: CDRDefinitionScheme

    def to_column_json(self) -> str:
        numbering_data = [{'domain': n.domain, 'position': n.query_position, 'label': n.label()} for n in
                          self.numbering]
        region_data = [r.to_data() for r in self.regions]
        antibody_numbering = {
            'scheme': str(self.cdr_definition),
            'numbering_scheme': str(self.numbering_scheme),
            'numbering': numbering_data,
            'regions': region_data
        }
        numbering_json = json.dumps(antibody_numbering)
        return numbering_json


def label_antibody_sequences(sequences: List[SeqRecord], numbering_scheme: NumberingScheme = NumberingScheme.KABAT,
                             cdr_definition: CDRDefinitionScheme = CDRDefinitionScheme.KABAT) -> List[List[AntibodyNumberMapping]]:
    mappings = _create_antibody_mappings(sequences, numbering_scheme)
    for mapping, sequence in zip(mappings, sequences):
        _annotate_sequence(sequence, mapping, numbering_scheme, cdr_definition)
    return mappings


def align_antibody_sequences(sequences: List[SeqRecord], numbering_scheme: NumberingScheme = NumberingScheme.KABAT,
                             cdr_definition: CDRDefinitionScheme = CDRDefinitionScheme.KABAT) -> AntibodyAlignmentResult:
    sequences = [SeqRecord(s.seq.ungap(), s.id, s.name, s.description) for s in sequences]
    mappings = label_antibody_sequences(sequences, numbering_scheme, cdr_definition)
    alignments = _do_align_antibody_sequences(sequences, mappings, numbering_scheme, cdr_definition)
    return alignments


def _do_align_antibody_sequences(sequences: List[SeqRecord],
                                 mapping: List[List[AntibodyNumberMapping]], numbering_scheme: NumberingScheme,
                                 cdr_definition: CDRDefinitionScheme) -> AntibodyAlignmentResult:
    labellings = set()
    for seq_map in mapping:
        for pos in seq_map:
            labellings.add(AntibodyNumber(pos.domain, pos.chain, pos.position, pos.insertion, None))

    labellings = sorted(labellings)
    input_positions = [0] * len(sequences)
    aligned_seqs = [""] * len(sequences)
    labelling_positions = [0] * len(labellings)
    start = True
    position = 0
    seqs = [s.seq for s in sequences]

    for label_no, label in enumerate(labellings):

        sequence_labels = [_find_in_mapping(map, label, input_position) for map, input_position in
                           zip(mapping, input_positions)]
        if all(sl is None or sl.query_position is None for sl in sequence_labels):
            diff = 0
            label_position = position
        else:
            if start:
                start = False
                n_residues_before = [
                    next((pos.query_position for pos in seq_map if pos.query_position and pos.matches(label)), 0) for
                    seq_map in mapping]
                n_prefix = max(n_residues_before)
                position += n_prefix
                for seq_no, sequence in enumerate(seqs):
                    padding = n_prefix - n_residues_before[seq_no]
                    input_positions[seq_no] = n_residues_before[seq_no]
                    aligned_seqs[seq_no] = "-" * padding + sequence[0:n_residues_before[seq_no]] + aligned_seqs[seq_no]

            diff = max(l.query_position - o for l, o in zip(sequence_labels, input_positions) if
                       l and l.query_position is not None)
            label_position = position + diff

        for seq_no, sequence in enumerate(seqs):

            sequence_label = sequence_labels[seq_no]
            last = input_positions[seq_no]
            if sequence_label and sequence_label.query_position is not None:
                input_positions[seq_no] = sequence_label.query_position + 1
                n_insert = sequence_label.query_position - last
                for x in range(n_insert):
                    aligned_seqs[seq_no] += sequence[last + x]
                aligned_seqs[seq_no] += '-' * (diff - n_insert)
                aligned_seqs[seq_no] += sequence[sequence_label.query_position]
            else:
                aligned_seqs[seq_no] += '-' * (diff + 1)

            assert (len(aligned_seqs[seq_no]) == label_position + 1)

        position = label_position + 1
        labelling_positions[label_no] = label_position

    n_residues_after = max(len(s) - p for s, p in zip(seqs, input_positions)) if seqs else 0;
    for seq_no, sequence in enumerate(seqs):
        pos = input_positions[seq_no]
        padding = n_residues_after - len(sequence) + pos
        aligned_seqs[seq_no] += sequence[pos:]
        aligned_seqs[seq_no] += '-' * padding

    aligned_records = [SeqRecord(id=s.id, name=s.name, seq=seq, annotations={'molecule_type': 'protein'}) for s, seq in
                       zip(sequences, aligned_seqs)]
    for init, align in zip(sequences, aligned_records):
        copy_features_to_gapped_sequence(init, align)

    numbering = [AntibodyNumber(n.domain, n.chain, n.position, n.insertion, p) for p, n in
                 zip(labelling_positions, labellings)]
    regions = _find_all_regions(numbering, numbering_scheme, cdr_definition)

    for index, m in enumerate(mapping):
        if not m:
            aligned_records[index] = None
    return AntibodyAlignmentResult(aligned_records, numbering, regions, numbering_scheme, cdr_definition)


def _find_in_mapping(mapping: Optional[List[AntibodyNumberMapping]], position: AntibodyNumber, input_position: int) -> \
Optional[AntibodyNumberMapping]:
    if not mapping:
        return None
    for m in mapping:
        if m.query_position is None:
            continue
        if m.query_position < input_position:
            continue
        antibody_number = m.to_antibody_number()
        if position < antibody_number:
            return None
        if antibody_number == position:
            return m
    return None


class AnarciDomain(NamedTuple):
    sequence_start: int
    sequence_end: int
    numbers: List[AntibodyNumberMapping]


def _create_antibody_mappings(sequences: List[SeqRecord], numbering_scheme: NumberingScheme) -> List[
    List[AntibodyNumberMapping]]:
    base = str(uuid.uuid4())
    in_file = 'seq_in_{}.fasta'.format(base)
    out_file = 'anarci_numbering_{}.txt'.format(base)
    sequences_to_file(in_file, sequences)

    lines: List[str] = []
    if sequences:
        command = AnarciCommandLine(cmd='ANARCI', sequence=in_file, scheme=str(numbering_scheme), outfile=out_file,
                                    restrict='ig')
        stdout, stderr = command()

        with open(out_file, encoding='utf8') as fh:
            lines = fh.readlines()

        for file in [in_file, out_file]:
            if os.path.exists(file):
                os.remove(file)

    mappings = []
    current_mapping = []
    current_domain = None
    domain_no = 0
    in_domain = False

    for line in lines:
        if line.startswith('//'):
            # print("start")
            mappings.append(current_mapping)
            current_domain = None
            current_mapping = []
        elif line.startswith('# Domain '):
            # print("Domain line {}"+line)
            terms = line.split(' ')
            domain_no = int(terms[2])
            in_domain = False
        elif line.startswith('#|') and not line.startswith('#|species'):
            # print("Info line {}"+line)
            items = line.split('|')
            current_domain = AnarciDomain(int(items[-3]), int(items[-2]), list())
            current_mapping.append(current_domain)
            assert not in_domain
            in_domain = True
        elif not line.startswith('#'):
            chain = line[0:1]
            assert chain == 'L' or chain == 'H'
            position = int(line[2:7])
            insertion = line[8:9].strip()
            if not insertion:
                insertion = None
            residue = line[10:11]
            current_domain.numbers.append(AntibodyNumberMapping(domain_no, chain, position, insertion, residue, None))

    assert len(mappings) == len(sequences)
    for mapping, record in zip(mappings, sequences):
        if not mapping:
            continue
        for domain in mapping:
            mapping_seq_arr = [m.residue for m in domain.numbers if m.residue != '-']
            mapping_seq = ''.join(mapping_seq_arr)
            assert mapping_seq in record.seq
            assert mapping_seq in record.seq[domain.sequence_start: domain.sequence_end + 1]

    mappings = [_match_to_sequence(s, m) for m, s in zip(mappings, sequences)]
    return mappings


def _match_to_sequence(record: SeqRecord, domains: List[AnarciDomain]) -> List[AntibodyNumberMapping]:
    domain_mappings = [_match_to_domain(record, d) for d in domains]
    return [m for dm in domain_mappings for m in dm]


def _match_to_domain(record: SeqRecord, domain: AnarciDomain) -> List[AntibodyNumberMapping]:
    mapping = domain.numbers
    match = ''.join([m.residue for m in mapping if m.residue != '-'])
    record_position = str(record.seq).find(match)
    assert record_position >= 0

    queue = collections.deque(mapping)
    mapping_to_record = []
    while queue:
        m = queue.popleft()
        if m.residue == '-':
            mapping_to_record.append(m)
        else:
            assert record[record_position] == m.residue
            mapping_to_record.append(
                AntibodyNumberMapping(m.domain, m.chain, m.position, m.insertion, m.residue, record_position))
            record_position += 1
    return mapping_to_record


def _find_all_regions(mapping: List[AntibodyNumber], numbering_scheme: NumberingScheme,
                      cdr_definition: CDRDefinitionScheme) -> List[ChainRegions]:
    def regions_in_domain(domain):
        filtered_mapping = [m for m in mapping if m.domain == domain]
        return _find_regions(filtered_mapping, numbering_scheme, cdr_definition)

    domains = set((m.domain for m in mapping))
    regions = [i for d in domains for i in regions_in_domain(d) if i]
    return regions


def _find_regions(mapping: List[AntibodyNumber], numbering_scheme: NumberingScheme,
                  cdr_definition: CDRDefinitionScheme) -> List[Optional[ChainRegions]]:
    # see CDR definitions in http://www.bioinf.org.uk/abs/info.html#cdrdef
    # This table is a little unclear as the end of H1 is not specified if the numbering is neither Kabat or Chothia
    # I have assumed that we use Chothia for everything except Kabat
    ###############
    # 23.03.11
    # IMGT_LEFRANC was added to correspond to cdr definitions in
    #  * Lefranc, Immunology Today 18, 509(1997)
    #  * Lefranc, The Immunologist 7, 132 - 136(1999)
    #  * Lefranc, et al.Dev.Comp.Immun. 27, 55 - 77(2003)
    # Lefranc CDR definitions are consistent with The Antibody Prediction Toolbox
    # at https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred and various tools from Charlotte Deane's lab.
    ###############

    # numbering_scheme numbers residues
    # cdr_definition assigns regions

    # the Wolfguy numbering is not supported
    assert numbering_scheme in [NumberingScheme.CHOTHIA, NumberingScheme.KABAT, NumberingScheme.IMGT,
                                NumberingScheme.MARTIN, NumberingScheme.AHO]
    assert cdr_definition in [CDRDefinitionScheme.CHOTHIA, CDRDefinitionScheme.KABAT, CDRDefinitionScheme.IMGT,
                              CDRDefinitionScheme.IMGT_LEFRANC]
    regions = list()

    if cdr_definition == CDRDefinitionScheme.KABAT:

        r = _find_chain_regions(mapping, 'L', 'L24', 'L34', 'L50', 'L56', 'L89', 'L97')
        regions.append(r)

        if numbering_scheme == CDRDefinitionScheme.KABAT:
            h1_end = 'H35B'
        else:
            h1_end = 'H35'

        r = _find_chain_regions(mapping, 'H', 'H31', h1_end, 'H50', 'H65', 'H95', 'H102')
        regions.append(r)

    elif cdr_definition == CDRDefinitionScheme.CHOTHIA:

        r = _find_chain_regions(mapping, 'L', 'L24', 'L34', 'L50', 'L56', 'L89', 'L97')
        regions.append(r)

        if numbering_scheme == NumberingScheme.KABAT:
            h35a = _find_antibody_number(mapping, 'H35A')
            h35a_present = h35a and h35a.position == 35 and h35a.insertion == 'A'
            h35b = _find_antibody_number(mapping, 'H35B')
            h35b_present = h35b and h35b.position == 35 and h35b.insertion == 'B'
            if not h35a_present and not h35b_present:
                h1_end = 'H32'
            elif h35a_present and h35b_present:
                h1_end = 'H34'
            elif h35a_present:
                h1_end = 'H33'
            else:
                raise ValueError()
        else:
            h1_end = 'H32'

        r = _find_chain_regions(mapping, 'H', 'H26', h1_end, 'H52', 'H56', 'H95', 'H102')
        regions.append(r)

    elif cdr_definition == CDRDefinitionScheme.IMGT:

        r = _find_chain_regions(mapping, 'L', 'L27', 'L32', 'L50', 'L51', 'L89', 'L97')
        regions.append(r)

        if numbering_scheme == NumberingScheme.KABAT:
            h1_end = 'H35B'
        else:
            h1_end = 'H33'
        r = _find_chain_regions(mapping, 'H', 'H26', h1_end, 'H51', 'H56', 'H93', 'H102')
        regions.append(r)

    elif cdr_definition == CDRDefinitionScheme.IMGT_LEFRANC:

        # mapping of numbering schemes to IMGT_LEFRANC CDR definitions based on
        # Dondelinger, et al. Front. Immunol. 9 (2018).
        if numbering_scheme == NumberingScheme.KABAT:
            r = _find_chain_regions(mapping, 'L', 'L27', 'L32', 'L50', 'L52', 'L89', 'L97')
            regions.append(r)
            r = _find_chain_regions(mapping, 'H', 'H26', 'H35B', 'H51', 'H57', 'H93', 'H102')
            regions.append(r)
        elif numbering_scheme == NumberingScheme.CHOTHIA:
            r = _find_chain_regions(mapping, 'L', 'L27', 'L32', 'L50', 'L52', 'L89', 'L97')
            regions.append(r)
            r = _find_chain_regions(mapping, 'H', 'H26', 'H33', 'H51', 'H57', 'H93', 'H102')
            regions.append(r)
        elif numbering_scheme == NumberingScheme.MARTIN:
            r = _find_chain_regions(mapping, 'L', 'L27', 'L32', 'L50', 'L52E', 'L89', 'L97')
            regions.append(r)
            r = _find_chain_regions(mapping, 'H', 'H26', 'H33', 'H51', 'H57', 'H93', 'H102')
            regions.append(r)
        elif numbering_scheme == NumberingScheme.AHO:
            r = _find_chain_regions(mapping, 'L', 'L27', 'L40', 'L58', 'L67', 'L107', 'L138')
            regions.append(r)
            r = _find_chain_regions(mapping, 'H', 'H27', 'H40', 'H58', 'H67', 'H107', 'H138')
            regions.append(r)
        elif numbering_scheme == NumberingScheme.IMGT:
            r = _find_chain_regions(mapping, 'L', 'L27', 'L38', 'L56', 'L65', 'L105', 'L117')
            regions.append(r)
            r = _find_chain_regions(mapping, 'H', 'H27', 'H38', 'H56', 'H65', 'H105', 'H117')
            regions.append(r)
        else:
            raise ValueError()
    else:
        raise ValueError()

    return regions


def _annotate_sequence(record: SeqRecord, number_mapping: List[AntibodyNumberMapping], numbering_scheme: NumberingScheme,
                       cdr_definition: CDRDefinitionScheme = None):
    def region_note(region_, name_):
        return ['antibody_label: {}'.format(name_),
                'antibody_domain: {}'.format(region_.domain),
                'antibody_chain: {}'.format(region_.chain),
                'cdr_definition: {}'.format(cdr_definition),
                'antibody_scheme: {}'.format(numbering_scheme)]

    mapping = [n.to_antibody_number() for n in number_mapping]
    all_regions = _find_all_regions(mapping, numbering_scheme, cdr_definition)
    for region in all_regions:

        start = region.start.query_position
        end = region.end.query_position
        start1 = region.cdr1_start.query_position
        end1 = region.cdr1_end.query_position
        start2 = region.cdr2_start.query_position
        end2 = region.cdr2_end.query_position
        start3 = region.cdr3_start.query_position
        end3 = region.cdr3_end.query_position
        chain = region.chain

        if not start1 or not end1 or not start2 or not end2 or not start3 or not end3:
            return

        if start <= start1:
            name = '{}FR1'.format(chain)
            fr1_feature = SeqFeature(FeatureLocation(start, start1), type='region',
                                     qualifiers={'region_name': name, 'note': region_note(region, name)})
            record.features.append(fr1_feature)

        if start1 <= end1 + 1:
            name = 'CDR-{}1'.format(chain)
            h1_feature = SeqFeature(FeatureLocation(start1, end1 + 1), type='region',
                                    qualifiers={'region_name': name, 'note': region_note(region, name)})
            record.features.append(h1_feature)

        if end1 + 1 <= start2:
            name = '{}FR2'.format(chain)
            fr1_feature = SeqFeature(FeatureLocation(end1 + 1, start2), type='region',
                                     qualifiers={'region_name': name, 'note': region_note(region, name)})
            record.features.append(fr1_feature)

        if start2 <= end2 + 1:
            name = 'CDR-{}2'.format(chain)
            h1_feature = SeqFeature(FeatureLocation(start2, end2 + 1), type='region',
                                    qualifiers={'region_name': name, 'note': region_note(region, name)})
            record.features.append(h1_feature)

        if end2 + 1 <= start3:
            name = '{}FR3'.format(chain)
            fr1_feature = SeqFeature(FeatureLocation(end2 + 1, start3), type='region',
                                     qualifiers={'region_name': name, 'note': region_note(region, name)})
            record.features.append(fr1_feature)

        if start3 <= end3 + 1:
            name = 'CDR-{}3'.format(chain)
            h1_feature = SeqFeature(FeatureLocation(start3, end3 + 1), type='region',
                                    qualifiers={'region_name': name, 'note': region_note(region, name)})
            record.features.append(h1_feature)

        if end3 + 1 <= end + 1:
            name = '{}FR4'.format(chain)
            fr1_feature = SeqFeature(FeatureLocation(end3 + 1, end + 1), type='region',
                                     qualifiers={'region_name': name, 'note': region_note(region, name)})
            record.features.append(fr1_feature)

    for num in mapping:
        if num.query_position is not None:
            num_feature = SeqFeature(FeatureLocation(num.query_position, num.query_position + 1), type='misc_feature',
                                     qualifiers={'feature_name': num.label(),
                                                 'note': ['antibody_number: {}'.format(num.label()),
                                                          'antibody_domain: {}'.format(num.domain),
                                                          'antibody_chain: {}'.format(num.chain),
                                                          'cdr_definition: {}'.format(cdr_definition),
                                                          'antibody_scheme: {}'.format(numbering_scheme)]})
            record.features.append(num_feature)


def numbering_and_regions_from_sequence(seq: SeqRecord) -> Optional[AntibodyAlignmentResult]:
    feature: SeqFeature
    domain_pattern = re.compile(r'^antibody_domain: (\d+)$')
    chain_pattern = re.compile(r'^antibody_chain: (\w+)$')
    cdr_pattern = re.compile(r'^cdr_definition: (\w+)$')
    scheme_pattern = re.compile(r'^antibody_scheme: (\w+)$')
    number_pattern = re.compile(r'^antibody_number: (\w+)$')
    label_pattern = re.compile(r'^([A-Z])(\d+)([A-Z]*)$')

    cdr_definition: Optional[CDRDefinitionScheme] = None
    numbering_scheme: Optional[NumberingScheme] = None
    numbering: List[AntibodyNumber] = []
    for feature in seq.features:
        if feature.type != 'misc_feature':
            continue
        if 'note' in feature.qualifiers:
            notes: List[str] = feature.qualifiers['note']
            domain: Optional[int] = None
            chain: Optional[str] = None
            numbering_label: Optional[str] = None

            if all((n.startswith('antibody_') or n.startswith('cdr_') for n in notes)):
                for note in notes:
                    if not cdr_definition:
                        match = cdr_pattern.match(note)
                        if match:
                            cdr_definition = match.group(1)
                            continue
                    if not numbering_scheme:
                        match = scheme_pattern.match(note)
                        if match:
                            numbering_scheme = match.group(1)
                            continue

                    match = domain_pattern.match(note)
                    if match:
                        domain = int(match.group(1))
                        continue
                    match = chain_pattern.match(note)
                    if match:
                        chain = match.group(1)
                        continue
                    match = number_pattern.match(note)
                    if match:
                        numbering_label = match.group(1)
                        continue

            if domain and chain and numbering_label and numbering_scheme:
                match = label_pattern.match(numbering_label)
                if match:
                    assert chain == match.group(1)
                    position = int(match.group(2))
                    insertion = match.group(3)
                    if not insertion:
                        insertion = None
                    number = AntibodyNumber(domain, chain, position, insertion, feature.location.start)
                    numbering.append(number)
    if not numbering:
        return None

    regions = _find_all_regions(numbering, numbering_scheme, cdr_definition)
    return AntibodyAlignmentResult([seq], numbering, regions, numbering_scheme, cdr_definition)


def _find_chain_regions(mapping: List[AntibodyNumber], chain: str, start1: str, end1: str,
                        start2: str, end2: str, start3: str, end3: str) -> Optional[ChainRegions]:
    start = _find_antibody_number(mapping, chain)
    end = _find_antibody_number(mapping, chain, last=True)

    if not start or not end:
        return None

    start1 = _find_antibody_number_from_str(mapping, start1)
    end1 = _find_antibody_number_from_str(mapping, end1)
    start2 = _find_antibody_number_from_str(mapping, start2)
    end2 = _find_antibody_number_from_str(mapping, end2)
    start3 = _find_antibody_number_from_str(mapping, start3)
    end3 = _find_antibody_number_from_str(mapping, end3)

    if not start1 or not end1 or not start2 or not end2 or not start3 or not end3:
        return None

    return ChainRegions(mapping[0].domain, chain, start, end, start1, end1, start2, end2, start3, end3)


def _find_antibody_number_from_str(mapping: List[AntibodyNumber], label: str) -> Optional[AntibodyNumber]:
    if "matcher" not in _find_antibody_number_from_str.__dict__:
        _find_antibody_number_from_str.matcher = re.compile(r'^([HL])(\d+)([A-Z]?)$')
    match = _find_antibody_number_from_str.matcher.match(label)
    assert match
    chain, position, insertion = match.groups()
    if not insertion:
        insertion = None
    return _find_antibody_number(mapping, chain, int(position), insertion)


def _find_antibody_number(mapping: List[AntibodyNumber], chain: str, position: Optional[int] = None,
                          insertion: Optional[str] = None, last: bool = None) -> Optional[AntibodyNumber]:
    last_match = None
    last_insert_match = None
    for num in mapping:
        if num.query_position is None:
            continue
        if num.chain == chain and num.position == position and insertion:
            last_insert_match = num
        if (num.chain == chain and num.position == position and num.insertion == insertion) or (
                num.chain == chain and position and num.position > position) or (
                num.chain == chain and position is None):
            match = num
            if num.chain == chain and position and num.position > position and last_insert_match:
                match = last_insert_match
            if not last:
                return match
            else:
                last_match = match
    return last_match

class Antibody():
    """
    Class to contain antibody characteristics and functionalities
    Originally developed for Antibody Developability Metrics Data Function
    """

    # Original hydropathy values from Kyte, Doolittle. JMolBio 157:105(1982) (shown in comments)
    # Normalized to [1,2] as described in Raybould, et al. PNAS 116(10):2019
    Raybould_Hydropathy = {'ALA': 1.70,  # 1.80
                           'CYS': 1.78,  # 2.50
                           'ASP': 1.11,  # -3.50
                           'GLU': 1.11,  # -3.50
                           'PHE': 1.81,  # 2.80
                           'GLY': 1.46,  # -0.40
                           'HIS': 1.14,  # -3.20
                           'ILE': 2.00,  # 4.50
                           'LYS': 1.07,  # -3.90
                           'LEU': 1.92,  # 3.80
                           'MET': 1.71,  # 1.90
                           'ASN': 1.11,  # -3.50
                           'PRO': 1.32,  # -1.60
                           'GLN': 1.11,  # -3.50
                           'ARG': 1.00,  # -4.50
                           'SER': 1.41,  # -0.80
                           'THR': 1.42,  # -0.70
                           'VAL': 1.97,  # 4.20
                           'TRP': 1.40,  # -0.90
                           'TYR': 1.36}  # -1.30

    # Raybould, et al., PNAS 116(10):2019.
    Raybould_Charge = {'ALA': 0.0,
                       'CYS': 0.0,
                       'ASP': -1.0,
                       'GLU': -1.0,
                       'PHE': 0.0,
                       'GLY': 0.0,
                       'HIS': 0.1,
                       'ILE': 0.0,
                       'LYS': 1.0,
                       'LEU': 0.0,
                       'MET': 0.0,
                       'ASN': 0.0,
                       'PRO': 0.0,
                       'GLN': 0.0,
                       'ARG': 1.0,
                       'SER': 0.0,
                       'THR': 0.0,
                       'VAL': 0.0,
                       'TRP': 0.0,
                       'TYR': 0.0}

    def __init__(self, record: SeqRecord, structure: Structure) -> None:

        self.record = record
        self.structure = structure

        # CDR vicinity related data
        self.cdr_vicinity_residues = None
        self.cdr_viinity_distances = None

        # dictionary to translated residue numbering to residue identity
        self.numbering_residue_dict = None

        # surface exposure
        self.residue_rasa = {}

        # salt bridges
        self.salt_bridges = None
        self.salt_bridge_residues = {'ARG': {'atoms': ['NE', 'NH1', 'NH2'],
                                             'cognates': ['ASP', 'GLU']},
                                     'ASP': {'atoms': ['OD1', 'OD2'],
                                             'cognates': ['ARG', 'LYS']},
                                     'GLU': {'atoms': ['OE1', 'OE2'],
                                             'cognates': ['ARG', 'LYS']},
                                     'LYS': {'atoms': ['NZ'],
                                             'cognates': ['ASP', 'GLU']}}

    def _find_salt_bridges(self) -> None:
        """
        Identify salt bridges in the antibody structure.

        Only ASP, GLU, ARG, and LYS are considered with terminal amines of ARG/LYS, intra-chain
        amine of ARG, and the acid oxygens of ASP/GLU.  If an amine and oxygen between two residues
        are within 3.2 Angstroms, it is considered a salt bridge.

        Uses class member salt_bridges dictionary to store identified salt bridges.
        Dictionary is keyed by the residue ID, with value equal to the residue id of the
        salt bridge partner residue, and None for no salt bridge
        """

        # store all residue ids and whether they are in a salt bridge or not
        self.salt_bridges = {}

        # storage for residue coordinates and atom origin mapping
        arg_lys_xyz = []
        arg_lys_map = []
        asp_glu_xyz = []
        asp_glu_map = []

        # parse the structure and capture residues that can participate in salt bridges
        for residue in self.structure.get_residues():
            residue_name = residue.get_resname()
            residue_id = \
                f'{residue.get_full_id()[2]}{str(residue.get_full_id()[3][1])}{residue.get_full_id()[3][2].strip()}'

            if residue_name not in self.salt_bridge_residues:
                self.salt_bridges[residue_id] = None
                continue

            xyz = []
            for atom in residue.get_atoms():
                if atom.get_id() in self.salt_bridge_residues[residue_name]['atoms']:
                    xyz.append(atom.get_coord())

            if residue_name in ['ARG', 'LYS']:
                arg_lys_xyz.extend(xyz)
                arg_lys_map.extend([residue_id] * len(xyz))
            else:
                asp_glu_xyz.extend(xyz)
                asp_glu_map.extend([residue_id] * len(xyz))

        atom_distances = euclidean_distance_matrix(np.array(arg_lys_xyz), np.array(asp_glu_xyz))
        bridge_atom_indices = np.nonzero(atom_distances <= 3.2)
        bridge_atom_pairs = set([(arg_lys_map[rk_idx], asp_glu_map[de_idx])
                                 for rk_idx, de_idx in zip(bridge_atom_indices[0], bridge_atom_indices[1])])

        for pair in bridge_atom_pairs:
            self.salt_bridges[pair[0]] = pair[1]
            self.salt_bridges[pair[1]] = pair[0]

        for residue_id in set(arg_lys_map + asp_glu_map):
            if residue_id not in self.salt_bridges:
                self.salt_bridges[residue_id] = None

        return

    def _is_salt_bridge(self, query_residue_id: str) -> bool:
        """
        Checks whether the passed residue is in a salt bridge.

        :param query_residue_id: str, residue ID of interest from the antibody structure
        :return: boolean, True if residue is in a salt-bridge, False if not
        """

        if self.salt_bridges is None:
            self._find_salt_bridges()

        if query_residue_id not in self.salt_bridges:
            raise ValueError()

        return self.salt_bridges[query_residue_id] is not None

    def _extract_residue_rasa(self) -> None:
        """
        Extract the RASA values for residues in the antibody record into the member dictionary self.residue_rasa
        Assumes internal SeqRecord contains calculated RASA values.
        """

        for feature in self.record.features:
            if feature.type == 'misc_feature':
                residue_id = feature.qualifiers['feature_name'][0]
                residue = feature.extract(self.record)
                for residue_feature in residue.features:
                    if residue_feature.qualifiers['feature_name'][0] == 'Relative Accessible Surface Area':
                        rasa = float(residue_feature.qualifiers['note'][2].split(' ')[1][:-1])
                        self.residue_rasa[residue_id] = rasa

        return

    def get_exposed_residues(self) -> List[Optional[int]]:
        """
        Get the index numbers of residues marked as exposed

        :return: list of integers referring to the exposed residue locations in the sequence
        """

        exposed_residues = []

        for feature in self.record.features:
            if 'note' in feature.qualifiers and \
               len(feature.qualifiers['note']) == 6 and \
               'RASA_exposed' in feature.qualifiers['note'][1]:

                exposed_residues.append(int(feature.location.start))

        return exposed_residues

    def _get_residue_rasa(self, query_residue_id: str) -> float:
        """
        Returns the Relative Accessible Surface Area of the passed residue

        :param query_residue_id: str, residue ID of interest from the antibody structure
        :return: float, the RASA value for the passed residue
        """

        if len(self.residue_rasa) == 0:
            self._extract_residue_rasa()

        return self.residue_rasa[query_residue_id]

    def create_numbering_residue_dict(self) -> List[Optional[Notification]]:
        """
        Creates a dictionary with keys being the antibody numbering reisdue ID
        and values the residue type (e.g., ALA, GLY, etc.)

        :return: list of notifications, if any
        """

        notifications = []

        try:
            self.numbering_residue_dict = dict(
                zip([f'{residue.get_full_id()[2]}{residue.get_full_id()[3][1]}{residue.get_full_id()[3][2].strip()}'
                     for residue in self.structure.get_residues()],
                [residue.get_resname() for residue in self.structure.get_residues()]))
        except Exception as ex:
            notifications.append(Notification(level=NotificationLevel.ERROR,
                                              title=f'Create Numbering_Residue Dictionary',
                                              summary=f'An unexpected error occurred\n{ex.__class__} - {ex}',
                                              details=f'{traceback.format_exc()}'))

        return notifications


    def compute_CDR_lengths(self) -> Tuple[dict, List[Optional[Notification]]]:
        """
        Computes the lengths of CDR annotated regions of the passed sequence

        :return: a dictionary of CDR lengths
        """

        CDR_lengths = {'CDR-H1': 0, 'CDR-H2': 0, 'CDR-H3': 0,
                       'CDR-L1': 0, 'CDR-L2': 0, 'CDR-L3': 0}
        notifications = []

        try:
            for feature in self.record.features:
                if feature.type == 'region':
                    if feature.qualifiers['region_name'][0] in CDR_lengths:
                        CDR_lengths[feature.qualifiers['region_name'][0]] = (
                            len(str(feature.location.extract(self.record).seq).replace('-', '')))
        except Exception as ex:
            notifications.append(Notification(level=NotificationLevel.ERROR,
                                              title=f'Compute CDR Lengths - Sequence Record ID {self.record.id}',
                                              summary=f'An unexpected error occurred\n{ex.__class__} - {ex}',
                                              details=f'{traceback.format_exc()}'))

        return (CDR_lengths, notifications)


    def compute_CDR_vicinity(self) -> List[Optional[Notification]]:
        """
        Identify CDR residues that are >= 7.5% RASA, and all other exposed residues within 4.0 Ang
        From Raybould, et al., PNAS 116(10):2019.

        :return: list of notifications, if any
        """

        notifications = []
        exposed_cdr_residues = []
        exposed_noncdr_residues = []
        cdr_vicinity_residues = None
        full_residue_distance_matrix = None

        try:
            for feature in self.record.features:
                if feature.type == 'region':
                    # partition into CDR and non-CDR residues
                    if feature.qualifiers['region_name'][0] in ['CDR-H1', 'CDR-H2', 'CDR-H3',
                                                                'CDR-L1', 'CDR-L2', 'CDR-L3']:
                        target_list = exposed_cdr_residues
                    else:
                        target_list = exposed_noncdr_residues

                    # find residues with RASA >= 7.5%
                    cdr = feature.extract(self.record)
                    for cdr_feature in cdr.features:
                        if cdr_feature.type == 'misc_feature':
                            if 'antibody_number' in cdr_feature.qualifiers['note'][0]:
                                residue_number = cdr_feature.qualifiers['feature_name'][0]
                                residue = cdr_feature.extract(cdr)
                                for residue_feature in residue.features:
                                    if residue_feature.qualifiers['feature_name'][0] == 'Relative Accessible Surface Area':
                                        rasa = float(residue_feature.qualifiers['note'][2].split(' ')[1][:-1])
                                        if rasa >= 7.5:
                                            target_list.append(residue_number)

                elif feature.type == 'misc_feature':
                    # find exposed anchor residues
                    # anchor residues defined for IMGT Lefranc numbering
                    # https://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html
                    # Lefranc, M.-P., The Immunologist, 7, 132-136 (1999).
                    if feature.qualifiers['feature_name'][0] in ['H26', 'H39', 'H55', 'H66', 'H104', 'H118',
                                                                 'L26', 'L39', 'L55', 'L66', 'L104', 'L118']:
                        residue_number = feature.qualifiers['feature_name'][0]
                        anchor = feature.extract(self.record)
                        for anchor_feature in anchor.features:
                            if anchor_feature.qualifiers['feature_name'][0] == 'Relative Accessible Surface Area':
                                rasa = float(anchor_feature.qualifiers['note'][2].split(' ')[1][:-1])
                                if rasa >= 7.5:
                                    exposed_cdr_residues.append(residue_number)  # add anchor residues to the cdr set

            # CDR vicinity is then any other residue with RASA >= 7.5%
            # and having a heavy atom within 4.0 Ang of a CDR residue
            cdr_xyz = []
            cdr_residue_id_map = []
            noncdr_xyz = []
            noncdr_residue_id_map = []

            for residue in self.structure.get_residues():
                residue_id = \
                    f'{residue.get_full_id()[2]}{str(residue.get_full_id()[3][1])}{residue.get_full_id()[3][2].strip()}'
                if residue_id in exposed_cdr_residues:
                    new_coords = [atom.get_coord() for atom in residue.get_atoms() if 'H' not in atom.get_name()]
                    cdr_xyz.extend(new_coords)
                    cdr_residue_id_map.extend([residue_id] * len(new_coords))
                elif residue_id in exposed_noncdr_residues:
                    new_coords = [atom.get_coord() for atom in residue.get_atoms() if 'H' not in atom.get_name()]
                    noncdr_xyz.extend(new_coords)
                    noncdr_residue_id_map.extend([residue_id] * len(new_coords))

            cdr_xyz = np.array(cdr_xyz)
            noncdr_xyz = np.array(noncdr_xyz)

            cdr_noncdr_distances = euclidean_distance_matrix(cdr_xyz, noncdr_xyz)
            cdr_adjacent_residues = set([noncdr_residue_id_map[i]
                                         for i, v in enumerate(np.min(cdr_noncdr_distances, axis=0)) if v <= 4.0])
            cdr_vicinity_residues = sorted(exposed_cdr_residues + list(cdr_adjacent_residues))

            # create a complete distance matrix for CDR vicinity residues
            # and reduce it to the minimum distances between all pairs of residues

            # cdr vs cdr
            cdr_distances = euclidean_distance_matrix(cdr_xyz, cdr_xyz)

            # reduce cdr vs non-cdr distances to only cdr vicinity
            drop_residues = [i for i, v in enumerate(noncdr_residue_id_map) if v not in cdr_adjacent_residues]
            cdr_noncdr_distances_reduced = np.delete(cdr_noncdr_distances, drop_residues, axis=1)
            noncdr_residue_id_map_reduced = [v for v in noncdr_residue_id_map if v in cdr_vicinity_residues]

            # reduce non-cdr xyz to cdr-vicinity and get distances
            noncdr_xyz_reduced = np.delete(noncdr_xyz, drop_residues, axis=0)
            noncdr_reduced_distances = euclidean_distance_matrix(noncdr_xyz_reduced, noncdr_xyz_reduced)

            # build up complete atomic distance matrix
            upper_distance_matrix = np.append(cdr_distances, cdr_noncdr_distances_reduced, axis=1)
            lower_distance_matrix = np.append(np.transpose(cdr_noncdr_distances_reduced), noncdr_reduced_distances, axis=1)
            full_distance_matrix = np.append(upper_distance_matrix, lower_distance_matrix, axis=0)
            full_residue_id_map = cdr_residue_id_map + noncdr_residue_id_map_reduced

            # reduce to residue distance matrix
            # each entry (i, j) is the minimum distance between heavy atoms
            # of residues i and j
            full_residue_distance_matrix = np.zeros(shape=(len(cdr_vicinity_residues), len(cdr_vicinity_residues)))
            for residue_index_i, residue_i in enumerate(cdr_vicinity_residues):
                for residue_index_j, residue_j in enumerate(cdr_vicinity_residues):
                    block_indices_i = [i for i, v in enumerate(full_residue_id_map) if v == residue_i]
                    block_indices_j = [j for j, v in enumerate(full_residue_id_map) if v == residue_j]
                    full_residue_distance_matrix[residue_index_i, residue_index_j] = \
                        np.min(full_distance_matrix[block_indices_i[0]:block_indices_i[-1],
                               block_indices_j[0]:block_indices_j[-1]])

        except Exception as ex:
            notifications.append(Notification(level=NotificationLevel.ERROR,
                                              title=f'Compute CDR Vicinity - Sequence Record ID {self.record.id}',
                                              summary=f'An unexpected error occurred\n{ex.__class__} - {ex}',
                                              details=f'{traceback.format_exc()}'))

        self.cdr_vicinity_residues = cdr_vicinity_residues
        self.cdr_vicinity_distances = full_residue_distance_matrix

        return notifications

    def compute_PSH(self) -> (Tuple)[float, List[Optional[Notification]]]:
        """
        Computes the Patches of Surface Hydrophobicity metric from Raybould, et al., PNAS 116(10):2019.

        :return: the PSH metric
        """

        notifications = []

        if self.cdr_vicinity_residues is None or self.cdr_vicinity_distances is None:
            vicinity_notifications = self.compute_CDR_vicinity()
            notifications.extend(vicinity_notifications)

        if self.numbering_residue_dict is None:
            notifications.extend(self.create_numbering_residue_dict())

        psh = 0

        for i in range(self.cdr_vicinity_distances.shape[0] - 1):
            for j in range(i + 1, self.cdr_vicinity_distances.shape[1]):
                if self.cdr_vicinity_distances[i, j] < 7.5:
                    residue_i = self.numbering_residue_dict[self.cdr_vicinity_residues[i]]
                    if residue_i in self.salt_bridge_residues:
                        if self._is_salt_bridge(self.cdr_vicinity_residues[i]):
                            HRi = self.Raybould_Hydropathy['GLY']
                        else:
                            HRi = self.Raybould_Hydropathy[self.numbering_residue_dict[self.cdr_vicinity_residues[i]]]
                    else:
                        HRi = self.Raybould_Hydropathy[self.numbering_residue_dict[self.cdr_vicinity_residues[i]]]

                    residue_j = self.numbering_residue_dict[self.cdr_vicinity_residues[j]]
                    if residue_j in self.salt_bridge_residues:
                        if self._is_salt_bridge(self.cdr_vicinity_residues[j]):
                            HRj = self.Raybould_Hydropathy['GLY']
                        else:
                            HRj = self.Raybould_Hydropathy[self.numbering_residue_dict[self.cdr_vicinity_residues[j]]]
                    else:
                        HRj = self.Raybould_Hydropathy[self.numbering_residue_dict[self.cdr_vicinity_residues[j]]]

                    psh += (HRi * HRj) / (self.cdr_vicinity_distances[i, j] ** 2)

        # comparisons to TAP showed PSH values ~ 0.5 * TAP values
        # I can only rationalize this as each pair of residues being
        # counted twice, rather than just the half-matrix of non-redundant
        # residue pairs
        return 2.0 * psh, notifications

    def compute_PPC(self) -> (Tuple)[float, List[Optional[Notification]]]:
        """
        Computes the Patches of Positive Charge metric from Raybould, et al., PNAS 116(10):2019.

        :return: the PPC metric
        """

        notifications = []

        if self.cdr_vicinity_residues is None or self.cdr_vicinity_distances is None:
            vicinity_notifications = self.compute_CDR_vicinity()
            notifications.extend(vicinity_notifications)

        if self.numbering_residue_dict is None:
            notifications.extend(self.create_numbering_residue_dict())

        ppc = 0

        for i in range(self.cdr_vicinity_distances.shape[0] - 1):
            for j in range(i + 1, self.cdr_vicinity_distances.shape[1]):
                if self.cdr_vicinity_distances[i, j] < 7.5:
                    residue_i = self.numbering_residue_dict[self.cdr_vicinity_residues[i]]
                    if residue_i in ['ARG', 'LYS', 'HIS']:
                        # Raybbould, et al 2019 describes setting charge for any residue involved
                        # in a salt-bridge to 0.  Comparison with online TAP tools
                        # suggests that this is not done.
                        # if self._is_salt_bridge(self.cdr_vicinity_residues[i]):
                        #     continue
                        # else:
                            QRi = abs(self.Raybould_Charge[self.numbering_residue_dict[self.cdr_vicinity_residues[i]]])
                    else:
                        continue

                    residue_j = self.numbering_residue_dict[self.cdr_vicinity_residues[j]]
                    if residue_j in ['ARG', 'LYS', 'HIS']:
                        # if self._is_salt_bridge(self.cdr_vicinity_residues[j]):
                        #     continue
                        # else:
                            QRj = abs(self.Raybould_Charge[self.numbering_residue_dict[self.cdr_vicinity_residues[j]]])
                    else:
                        continue

                    ppc += (QRi * QRj) / (self.cdr_vicinity_distances[i, j] ** 2)

        # comparisons to TAP showed PPC values ~ 0.5 * TAP values
        # (once salt-bridge check was disabled)
        # I can only rationalize this as each pair of residues being
        # counted twice, rather than just the half-matrix of non-redundant
        # residue pairs
        return 2.0 * ppc, notifications

    def compute_PNC(self) -> (Tuple)[float, List[Optional[Notification]]]:
        """
        Computes the Patches of Negative Charge metric from Raybould, et al., PNAS 116(10):2019.

        :return: the PNC metric
        """

        notifications = []

        if self.cdr_vicinity_residues is None or self.cdr_vicinity_distances is None:
            vicinity_notifications = self.compute_CDR_vicinity()
            notifications.extend(vicinity_notifications)

        if self.numbering_residue_dict is None:
            notifications.extend(self.create_numbering_residue_dict())

        pnc = 0

        for i in range(self.cdr_vicinity_distances.shape[0] - 1):
            for j in range(i + 1, self.cdr_vicinity_distances.shape[1]):
                if self.cdr_vicinity_distances[i, j] < 7.5:
                    residue_i = self.numbering_residue_dict[self.cdr_vicinity_residues[i]]
                    if residue_i in ['ASP', 'GLU']:
                        # Raybbould, et al 2019 describes setting charge for any residue involved
                        # in a salt-bridge to 0.  Comparison with online TAP tools
                        # suggests that this is not done.
                        # if self._is_salt_bridge(self.cdr_vicinity_residues[i]):
                        #     continue
                        # else:
                            QRi = abs(self.Raybould_Charge[self.numbering_residue_dict[self.cdr_vicinity_residues[i]]])
                    else:
                        continue

                    residue_j = self.numbering_residue_dict[self.cdr_vicinity_residues[j]]
                    if residue_j in ['ASP', 'GLU']:
                        # if self._is_salt_bridge(self.cdr_vicinity_residues[j]):
                        #     continue
                        # else:
                            QRj = abs(self.Raybould_Charge[self.numbering_residue_dict[self.cdr_vicinity_residues[j]]])
                    else:
                        continue

                    pnc += (QRi * QRj) / (self.cdr_vicinity_distances[i, j] ** 2)

        # comparisons to TAP showed PNC values ~ 0.5 * TAP values
        # (once salt-bridge check was disabled)
        # I can only rationalize this as each pair of residues being
        # counted twice, rather than just the half-matrix of non-redundant
        # residue pairs
        return 2.0 * pnc, notifications

    def compute_SFvCSP(self) -> (Tuple)[float, List[Optional[Notification]]]:
        """
        Computes the Structural Fv Charge Symmetry Parameter metric from Raybould, et al., PNAS 116(10):2019.

        :return: the SFvCSP metric
        """

        notifications = []

        if self.numbering_residue_dict is None:
            notifications.extend(self.create_numbering_residue_dict())

        Qsum_H = 0  # sum of heavy chain charges
        Qsum_L = 0  # sum of light chain charges

        for res_id, res_name in self.numbering_residue_dict.items():
            try:
                if self._get_residue_rasa(res_id) >= 7.5:
                    if res_id.startswith('H'):
                        Qsum_H += self.Raybould_Charge[res_name]
                    elif res_id.startswith('L'):
                        Qsum_L += self.Raybould_Charge[res_name]
                    else:
                        notifications.append(Notification(level=NotificationLevel.ERROR,
                                                          title=f'Compute Charge Symmetry {self.record.id}',
                                                          summary='Expected chain ID not found.',
                                                          details='Antibody structures are expected to have two chains - ' +
                                                                  'H (heavy) and L (light) chain. ' +
                                                                  'One or both are missing.'))
                        Qsum_H = None
                        Qsum_L = None
                        break
            except KeyError:
                notifications.append(Notification(level=NotificationLevel.WARNING,
                                                  title=f'Compute Charge Symmetry {self.record.id}',
                                                  summary='Structure - sequence mismatch.',
                                                  details='The paired antibody structure and sequence are mismatched. ' +
                                                           'Ensure that the antibody sequence column is that generated ' +
                                                           'by the Antibody Prediction data function.  Mismatched ' +
                                                           'structure and sequence will result in inaccurate metrics.'))

        if Qsum_H is None or Qsum_L is None:
            SFvCSP = None
        else:
            SFvCSP = Qsum_H * Qsum_L

        return float(SFvCSP), notifications


@dataclass
class AntibodySequencePair:
    H: str
    L: str


def extract_ABB2_numbering(ab_numberings: list[dict], ab_sequence_pairs: list[AntibodySequencePair]) \
        -> Tuple[list[str], dict[ANTIBODY_NUMBERING_COLUMN_PROPERTY, str]]:
    """
    Convert the ABodyBuilder 2 numbers to an internal mapping representation
    that can be used for processing numbering information

    :param ab_numberings: a list containing the dictionary of chain numberings
                          which is the .numbered_sequences element of a predicted
                          antibody structure from ABodyBuilder 2
    :param ab_sequence_pairs:  list of AntibodySequencePairs for antibody sequences corresponding
                               to ab_numberings - expected to have names 'H' and 'L' corresponding to
                               Heavy and Light chain sequences, resp.
    :return:  A tuple consisting of:
                a list of base64 encoded GenBank records as strings
                a dictionary with keys as ANTIBODY_NUMBERING_COLUMN_PROPERTY and
                    values being a string of the column property JSON
    """

    mappings = []
    for sequences, numbering in zip(ab_sequence_pairs, ab_numberings):
        mapping = []
        for domain_idx, chain in enumerate(['H', 'L']):
            domain = numbering[chain]
            numbers = []
            last_position = 0
            for residue_idx, residue in enumerate(domain):
                position = residue[0][0]
                if position != last_position and position != last_position + 1:
                    for gap_position in range(last_position + 1, position):
                        numbers.append(AntibodyNumberMapping(domain=domain_idx + 1, chain=chain,
                                                             position=gap_position, insertion=None,
                                                             residue='-', query_position=None))
                insertion = residue[0][1].strip()
                if not insertion:
                    insertion = None
                numbers.append(AntibodyNumberMapping(domain = domain_idx + 1, chain = chain,
                                                     position = residue[0][0], insertion = insertion,
                                                     residue = residue[1], query_position = None))
                last_position = position

            mapping.append(AnarciDomain(sequence_start = 0, sequence_end = len(sequences.__dict__[chain]) - 1,
                                        numbers = numbers))

        mappings.append(mapping)

    ab_sequences = [SeqRecord(Seq(seq_pair.H + seq_pair.L)) for seq_pair in ab_sequence_pairs]

    mappings2 = [_match_to_sequence(s, m) for s, m in zip(ab_sequences, mappings)]
    for mapping, sequence in zip(mappings2, ab_sequences):
        _annotate_sequence(sequence, mapping, NumberingScheme.IMGT, CDRDefinitionScheme.IMGT_LEFRANC)
    align_information = _do_align_antibody_sequences(ab_sequences, mappings2,
                                                     NumberingScheme.IMGT, CDRDefinitionScheme.IMGT_LEFRANC)

    output_sequences = align_information.aligned_sequences
    numbering_json = align_information.to_column_json()

    rows = [sequence_to_genbank_base64_str(s) for s in output_sequences]
    properties = {ANTIBODY_NUMBERING_COLUMN_PROPERTY: numbering_json}

    return rows, properties
