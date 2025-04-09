from Bio.Restriction import RestrictionBatch, AllEnzymes, CommOnly
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.SeqRecord import SeqRecord

from df.bio_helper import column_to_sequences
from df.data_transfer import DataFunction, DataFunctionRequest, DataFunctionResponse, ColumnData, \
    DataType, string_list_input_field
from ruse.bio.bio_data_table_helper import sequence_to_genbank_base64_str
from ruse.bio.bio_util import is_dna_record


def search_sequence(rec: SeqRecord, rb: RestrictionBatch) -> SeqRecord:
    if not is_dna_record(rec):
        raise ValueError("Not a nucleotide sequence")
    seq = rec.seq.back_transcribe if 'U' in rec.seq else rec.seq
    matches = rb.search(seq)
    record = SeqRecord(seq, rec.id)
    for (enzyme, positions) in matches.items():
        if len(positions) == 0:
            continue
        name = str(enzyme)
        for position in positions:
            feature = SeqFeature(FeatureLocation(position - 1, position), type='misc_feature',
                                 qualifiers={'note': [f'enzyme: {name}',
                                                      f'sequence: {enzyme.elucidate()}']})
            record.features.append(feature)
    return record


class EnzymeRestriction(DataFunction):
    """
    Performs restriction analysis using codes from REBASE
    """

    def execute(self, request: DataFunctionRequest) -> DataFunctionResponse:
        enzyme_names = string_list_input_field(request, 'enzymes')
        if not enzyme_names:
            raise ValueError
        if 'All' in enzyme_names:
            rb = AllEnzymes
        elif 'Common' in enzyme_names:
            rb = CommOnly
        else:
            for name in enzyme_names:
                try:
                    AllEnzymes.get(name)
                except ValueError:
                    raise ValueError(f'{name} is not a known restriction enzyme')
            rb = RestrictionBatch(enzyme_names)

        input_column = next(iter(request.inputColumns.values()))
        input_sequences = column_to_sequences(input_column)
        output_sequences = [None if s is None else search_sequence(s, rb) for s in input_sequences]

        rows = [sequence_to_genbank_base64_str(s) for s in output_sequences]
        output_column = ColumnData(name=f'Digested {input_column.name}', dataType=DataType.BINARY,
                                   contentType='chemical/x-genbank', values=rows)
        output_column.insert_nulls(input_column.missing_null_positions)
        response = DataFunctionResponse(outputColumns=[output_column])
        return response
