from io import StringIO
from math import sqrt
import os
from typing import Any

from Bio import SeqRecord
from Bio.PDB import Atom as BioAtom
from Bio.PDB import PDBIO, PDBParser, Structure
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.SeqUtils import seq1 as amino3to1

from df.bio_helper import column_to_sequences, column_to_structures, generate_color_gradient
from df.data_transfer import ColumnData, DataFunctionRequest, DataFunctionResponse, DataFunction, DataType, \
    string_input_field, boolean_input_field, double_input_field, input_field_to_column, \
    Notification, NotificationLevel

from ruse.bio.bio_data_table_helper import sequence_to_genbank_base64_str

from openmm.app import Modeller, PDBFile


class ProteinRASA(DataFunction):
    """
    Class containing functions for computing relative solvent accessible surface area (RASA) of a given protein
    Intention is to compute RASA as closely as possible to that produced by
    The Antibody Prediction Toolbox (https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred)
    and described in Raybould, et al. PNAS 116(10), 2019.
    """

    def neighbors_to_alanine(self, trimer, terminus=None):
        alanine_atoms = ['N', 'H', 'CA', 'C', 'O', 'CB', 'HA']

        residues = [residue for residue in trimer.get_residues()]

        target_indices = []
        if terminus == 'N':
            if residues[1].resname != 'ALA':
                target_indices = [1]
        elif terminus == 'C':
            if residues[0].resname != 'ALA':
                target_indices = [0]
        else:
            target_indices = [index for index in [0, 2] if residues[index].resname != 'ALA']

        for target_index in target_indices:
            if residues[target_index].resname != 'GLY':
                residues[target_index].child_dict = \
                    {k: v for k, v in residues[target_index].child_dict.items() if k in alanine_atoms}
                residues[target_index].child_list = \
                    [a for a in residues[target_index].child_list if a.fullname.strip() in alanine_atoms]
            else:
                # determine which hydrogen to convert to CB
                trimer.atom_to_internal_coordinates()
                if residues[target_index].internal_coord.pick_angle('O:C:CA:HA2').angle > 0:
                    target_ha = 'HA2'
                    nontarget_ha = 'HA3'
                else:
                    target_ha = 'HA3'
                    nontarget_ha = 'HA2'

                # determine coordinates for new CB atom
                ca_coord = residues[target_index]['CA'].get_coord()
                h_coord = residues[target_index][target_ha].get_coord()
                ca_h = h_coord - ca_coord
                ca_h_unit = ca_h / sqrt(sum(ca_h ** 2))
                cb_coord = ca_coord + ca_h_unit * 1.5  # CA-CB bond length in alanine is 1.5 angstroms

                # create the new CB atom
                new_cb = BioAtom.Atom('CB', cb_coord, 1, 1.0, ' ', ' CB ',
                                      residues[target_index][target_ha].get_serial_number(), element='C')
                residues[target_index].detach_child(target_ha)
                residues[target_index].add(new_cb)

                # drop the extra hydrogen for openMM
                residues[target_index].detach_child(nontarget_ha)

            residues[target_index].resname = 'ALA'

        # convert BioPython Structure to OpenMM PDBFile
        io = PDBIO()
        io.set_structure(trimer)
        output = StringIO()
        io.save(output)
        output.seek(0)
        pdb = PDBFile(output)  # create OpenMM object

        # use OpenMM Modeler to add hydrogens
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens()

        # convert from OpenMM PDBFile to BioPython Structure
        newOutput = StringIO()
        pdb.writeFile(modeller.topology, modeller.positions, newOutput)
        newOutput.seek(0)

        # OpenMM writes the end of the PDB file as "END\n" and BioPython expects
        # "END   \n"  - adjust output of OpenMM to avoid unrecognized filed warnings
        offset = newOutput.seek(0, os.SEEK_END)
        newOutput.seek(offset - 4)
        newOutput.write('END   \n')
        newOutput.seek(0)

        # read as BioPython structure
        parser = PDBParser()
        new_trimer = parser.get_structure("trimer", newOutput)

        return new_trimer

    def run_ProteinRASA(self, structures: list[Structure], sequences: list[SeqRecord],
                        rasa_cutoff: float, exposed_color: str, label_all: bool, start_color: str, end_color: str) \
            -> tuple[dict[str, Any], list[Notification]]:
        """
        Predict Relative Accessible Surface Area from structure and return data for output columns

        :structures:  list of BioPython Structure objects containing the protein structures of interest
        :param sequences:  A list of BioPython SeqRecords containing an Antibody sequence.
                           Expected to be consistent with the sequences in `structures`
        :param rasa_cutoff: Cutoff value, RASA greater than or equal to determines Solvent Exposed status
        :param exposed_color: String - hex string color for exposed residues
        :param label_all:  Boolean - if true, all residues in the sequence will be given a RASA annotations
        :param start_color: String - hex string color defining the starting color in a color gradient
                                     for the lowest RASA of 0%
        :param end_color: String - hex string color defining the end color in a color gradient
                                   for the highest RASA of 1000%
        :return:  A tuple consisting of:
                     A dictionary with keys being strings with suggested column names and values of dictionaries.
                     The values for this dictionary is another dictionary having two possible keys:
                        'values' - containing the data values for the column
                        'properties' - containing property values to be applied to the column
        """

        if label_all:
            RASA_color_gradient = generate_color_gradient(start_color, end_color, 26)

        # setup BioPython tools
        sr = ShrakeRupley()
        sbld = StructureBuilder()

        # output and notification storage
        output_sequences = []
        null_notification_rows = []
        matching_notification_rows = []
        notifications = []
        SR_notifications = {}
        trimer_notification_rows = []

        # process each structure
        for index, (structure, sequence) in enumerate(zip(structures, sequences)):
            if not (structure and sequence):
                # accumulate rows with errors and just send one notification at the end
                output_sequences.append(None)
                null_notification_rows.append(index + 1)
                continue

            # create a mapping between the structure residues and potentially gapped sequence residues
            sequence_number_mapping = {}
            structure_residues = structure.get_residues()

            try:
                for sequence_residue_number, sequence_residue in enumerate(sequence.seq):
                    if sequence_residue != '-':
                        structure_residue = next(structure_residues)
                        if sequence_residue != amino3to1(structure_residue.resname):
                            raise Exception
                        sequence_number_mapping[structure_residue.get_full_id()] = sequence_residue_number
            except Exception as ex:
                # accumulate rows with errors and just send one notification at the end
                matching_notification_rows.append(index + 1)
                continue

            # compute RASA for each residue/atom in context of the intact protein
            try:
                sr.compute(structure, level='A')
            except Exception as ex:
                # accumulate rows with errors and just send one notification at the end
                SR_notifications[index + 1] = ex
                output_sequences.append(sequence)
                continue

            success = True  # for scoping outside loop

            for substructure in structure.get_chains():
                residues = [residue for residue in substructure.get_residues()]

                for residue_index, residue in enumerate(residues):
                    # compute RASA for each residue isolated from the protein structure flanked by alanines
                    sbld.init_structure('trimer_structure')
                    sbld.init_model('trimer_model')
                    sbld.init_chain('A')
                    chain = sbld.structure['trimer_model']['A']
                    terminus = None  # is this residue an N- or C-terminus
                    target_index = 1

                    if residue_index == 0:
                        terminus = 'N'
                        target_index = 0
                    else:
                        chain.add(residues[residue_index - 1].copy())

                    chain.add(residues[residue_index].copy())

                    if residue_index == len(residues) - 1:
                        terminus = 'C'
                    else:
                        chain.add(residues[residue_index + 1].copy())

                    # mutate the flanking residues to alanine
                    trimer = self.neighbors_to_alanine(sbld.get_structure(), terminus)
                    trimer_residue = trimer[0]['A'].child_list[target_index]

                    try:
                        sr.compute(trimer, level='A')
                    except Exception as ex:
                        success = False
                        break

                    in_context_sa = sum([atom.sasa for atom in residue.get_atoms()])
                    trimer_reference_sa = sum([atom.sasa for atom in trimer_residue.get_atoms()])

                    if trimer_reference_sa == 0.0:
                        # if the reference is zero, most likely hydrogens are missing from structure
                        # glycine will then have zero reference because of its one non-backbone hydrogen
                        # this will most likely never happen
                        rasa = 0.0
                    else:
                        rasa = round(in_context_sa / trimer_reference_sa * 100, 1)

                    # create annotations
                    if label_all:
                        feature = SeqFeature(FeatureLocation(sequence_number_mapping[residue.get_full_id()],
                                                             sequence_number_mapping[residue.get_full_id()] + 1),
                                             type='misc_feature',
                                             qualifiers={'feature_name': 'Relative Accessible Surface Area',
                                                         'note': ['RASA',
                                                                  'glysade_annotation_type: RASA',
                                                                  f'RASA: {rasa}%',
                                                                  f'Residue ID:  {residue.resname}',
                                                                  f'ld_style:{{"color": "{RASA_color_gradient[int(rasa // 5)]}", "shape": "rounded-rectangle"}}',
                                                                  'ld_track: RASA']})
                        sequence.features.append(feature)

                    if rasa >= rasa_cutoff:
                        feature = SeqFeature(FeatureLocation(sequence_number_mapping[residue.get_full_id()],
                                                             sequence_number_mapping[residue.get_full_id()] + 1),
                                             type='misc_feature',
                                             qualifiers={'feature_name': 'Solvent Exposed Residue',
                                                         'note': [f'Exposed >= {rasa_cutoff}%',
                                                                  'glysade_annotation_type: RASA_exposed',
                                                                  f'RASA: {rasa}%',
                                                                  f'Residue ID:  {residue.resname}',
                                                                  f'ld_style:{{"color": "{exposed_color}", "shape": "rounded-rectangle"}}',
                                                                  'ld_track: RASA_exposed']})
                        sequence.features.append(feature)

                if not success:
                    break

            if not success:
                trimer_notification_rows.append(index + 1)

            output_sequences.append(sequence)

        # construct notifications
        if len(null_notification_rows) != 0:
            notifications.append(Notification(level=NotificationLevel.ERROR,
                                              title='Relative Accessible Surface Area',
                                              summary='Error for structure or sequence in Row(s) ' +
                                                      ','.join([str(row) for row in null_notification_rows]) +
                                                      '.\nNull values found.',
                                              details='Either the structure or sequence were null.'))

        if len(matching_notification_rows) != 0:
            notifications.append(Notification(level=NotificationLevel.ERROR,
                                              title='Relative Accessible Surface Area',
                                              summary='Error for structure and sequence in Row(s) ' +
                                                      ','.join([str(row) for row in matching_notification_rows]) +
                                                      '.',
                                              details='Sequence in structure do not match those in sequence column.'))

        if len(SR_notifications) != 0:
            notifications.append(Notification(level=NotificationLevel.ERROR,
                                              title='Relative Accessible Surface Area',
                                              summary='Error for structure in Row(s) ' +
                                                      ','.join([str(index) for index, ex in SR_notifications.items()]) +
                                                      '.\n',
                                              details='Shrake-Rupley calculation failed for complete structure.'))

        if len(trimer_notification_rows) != 0:
            notifications.append(Notification(level=NotificationLevel.ERROR,
                                              title='Relative Accessible Surface Area',
                                              summary='Error for structure in Row(s) ' +
                                                      ','.join([str(row) for row in trimer_notification_rows]) +
                                                      '.\n',
                                              details='Shrake-Rupley calculation failed for some residues.'))

        # construct output column
        rows = [sequence_to_genbank_base64_str(s) for s in output_sequences]
        output_column_data = {'Relative Accessible Surface Area Annotations': {'values': rows}}

        return output_column_data, notifications

    def execute(self, request: DataFunctionRequest) -> DataFunctionResponse:

        # get settings from UI
        structure_column = input_field_to_column(request, 'uiStructureColumn')
        structures, notifications = column_to_structures(structure_column)

        sequence_column = input_field_to_column(request, 'uiSequenceColumn')
        sequences = column_to_sequences(sequence_column)

        rasa_cutoff = round(double_input_field(request, 'uiRASACutoff'), 1)
        exposed_color = string_input_field(request, 'uiExposedColor')

        label_all = boolean_input_field(request, 'uiLabelAll')
        start_color = string_input_field(request, 'uiStartColor')
        end_color = string_input_field(request, 'uiEndColor')

        output_column_data, rasa_notifications = \
            self.run_ProteinRASA(structures, sequences, rasa_cutoff, exposed_color, label_all, start_color, end_color)
        notifications.extend(rasa_notifications)

        output_column = ColumnData(name=f'Relative Accessible Surface Area Annotations',
                                   dataType=DataType.BINARY,
                                   contentType='chemical/x-genbank',
                                   values=output_column_data['Relative Accessible Surface Area Annotations']['values'])

        return DataFunctionResponse(outputColumns=[output_column], notifications=notifications)
