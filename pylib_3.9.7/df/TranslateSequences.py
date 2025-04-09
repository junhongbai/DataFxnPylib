from copy import deepcopy

from Bio.Data import CodonTable
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from df.bio_helper import column_to_sequences, sequences_to_column
from df.data_transfer import DataFunction, DataFunctionRequest, DataFunctionResponse, string_input_field
from ruse.bio.bio_util import is_dna_record


def translate(rec: SeqRecord, codon_table_name: str, init_site_method: str = None) -> SeqRecord:
    if is_dna_record(rec):
        init_sequence = deepcopy(rec)

        # reduce sequence based on initiation site
        if init_site_method == 'ATG':
            idx = init_sequence.seq.upper().find('ATG')
            if idx < 0:
                init_sequences = None
            else:
                init_sequence = init_sequence[idx:]
        elif init_site_method == 'table':
            codon_table = CodonTable.unambiguous_dna_by_name[codon_table_name]
            init_codons = codon_table.start_codons

            idx = [v for v in [init_sequence.seq.upper().find(codon) for codon in init_codons] if v != -1]
            if len(idx) == 0:
                init_sequence = None
            else:
                init_sequence = init_sequence[min(idx):]

        return init_sequence.translate(codon_table_name)

    # create single naive DNA sequence- maybe should error here instead
    s = rec.seq
    codon_table = CodonTable.unambiguous_dna_by_name[codon_table_name]
    mapping = codon_table.back_table
    codons = [mapping[r] if r in mapping else '-' for r in s]
    s = Seq(''.join(codons))
    rec = SeqRecord(s, rec.id, rec.name)

    return rec


class TranslateSequences(DataFunction):
    """
    Translates DNA in the input column to an output protein column.  If the input column is a protein sequence will
    create a naive DNA sequence in the output
    """

    def execute(self, request: DataFunctionRequest) -> DataFunctionResponse:
        input_column = next(iter(request.inputColumns.values()))
        input_sequences = column_to_sequences(input_column)

        codon_table_name = string_input_field(request, 'codonTableName', 'Standard')

        init_site_method = string_input_field(request, 'initMethod')

        output_sequences = [None if s is None else translate(s, codon_table_name, init_site_method)
                            for s in input_sequences]
        output_column = sequences_to_column(output_sequences, f'Translated {input_column.name}',
                                            genbank_output=False)
        response = DataFunctionResponse(outputColumns=[output_column])

        return response
