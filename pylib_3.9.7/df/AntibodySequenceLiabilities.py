from collections import defaultdict
import traceback
from typing import Optional, List

from df.bio_helper import column_to_sequences
from df.data_transfer import ColumnData, DataFunctionRequest, DataFunctionResponse, DataFunction, DataType, \
     input_field_to_column, string_list_input_field, string_input_field
from df.data_transfer import Notification, NotificationLevel

from ruse.bio.antibody import Antibody
from ruse.bio.bio_data_table_helper import sequence_to_genbank_base64_str
from ruse.bio.bio_util import find_regex_matches

from Bio import SeqRecord
from Bio.PDB.Structure import Structure
from Bio.SeqFeature import SeqFeature, FeatureLocation


class AntibodySequenceLiabilities(DataFunction):
    """
    Port of C# Data Function to search for defined antibody protein sequence liabilities
    """

    # Liability names and patterns as of 24.08.16
    SequenceLiabilityPatterns = {'Asparagine Deamidation': '(?=(N[GSTNH]))',
                                 'Aspartic Acid Fragmentation': 'DP',
                                 'Aspartic Acid Isomerization': '(?=(D[GSTDH]))',
                                 'Cysteine': 'C',
                                 'Lysine N-glycation': '(?<=(K[DE]|[DE]K))',
                                 'N-glycosylation': 'N[^P][TS]',
                                 'N-terminal Glutamic Acid': '^E',
                                 'Oxidation': '[MW]'}

    @staticmethod
    def run_sequence_liabilities(ab_sequences: list[SeqRecord], patterns: dict[str, str],
                                 output_column_name: str, color: str) \
            -> tuple[dict[str, list], list[Optional[Notification]]]:

        column_data = defaultdict(list)
        column_data[output_column_name] = ab_sequences.copy()
        liability_notifications = {}

        for name, pattern in patterns.items():
            for row_index, ab_sequence in enumerate(ab_sequences):
                if ab_sequence is None:
                    column_data[name].append(None)
                    continue

                ab = Antibody(ab_sequence, Structure(None))
                try:
                    matches = find_regex_matches(pattern=pattern, sequence=ab_sequence.seq)
                except Exception as ex:
                    liability_notifications[f'Row {row_index + 1}'] = (ex, f'{traceback.format_exc()}')
                    column_data[name].append(None)
                    continue

                if len(matches) == 0:
                    column_data[name].append(False)
                else:
                    exposed_residues = ab.get_exposed_residues()  # no exposed residues is interpreted as no RASA info
                    liability_found = len(exposed_residues) == 0
                    for match in matches:
                        match_in_range = False
                        if len(exposed_residues) != 0:
                            for liability_location in range(match[0], match[1]):
                                if liability_location in exposed_residues:
                                    liability_found = True
                                    match_in_range = True
                                    break
                                else:
                                    match_in_range = False
                        else:
                            match_in_range = True  # fake the match in range when no RASA

                        if match_in_range:
                            feature = SeqFeature(FeatureLocation(match[0], match[1]),
                                                 type='misc_feature',
                                                 qualifiers={'feature_name': 'Antibody Sequence Liability',
                                                             'note': [name,
                                                                      'glysade_annotation_type: sequence_liability',
                                                                      f'ld_style:{{"color": "{color}", "shape": "rounded-rectangle"}}',
                                                                      'ld_track: sequence_liability']})
                            ab_sequence.features.append(feature)

                    column_data[name].append(liability_found)

        # compile notifications
        notifications = []
        if len(notifications) != 0:
            notifications.append(Notification(level=NotificationLevel.ERROR,
                                              title='Antibody Sequence Liabilities',
                                              summary=f'Error for IDs ' +
                                                      ','.join([row_index
                                                                for row_index in liability_notifications.keys()]) +
                                                      '.\n',
                                              details=''.join([f'{row_index}\n{ex.__class__} - {ex}\n' +
                                                               f'{details}\n\n'
                                                               for row_index, (ex, details)
                                                               in liability_notifications.items()])))

        return column_data, notifications

    def execute(self, request: DataFunctionRequest) -> DataFunctionResponse:

        sequence_column = input_field_to_column(request, 'sequenceColumn')
        ab_sequences = column_to_sequences(sequence_column)

        output_column_name = string_input_field(request, 'outputColumn')
        liability_patterns = {}
        for name_pattern in string_list_input_field(request, 'patterns'):
            tokens = name_pattern.split(',')
            liability_patterns[tokens[0]] = tokens[1]

        if len(liability_patterns) == 0:
            notifications = [Notification(level=NotificationLevel.WARNING,
                                          title='Antibody Liabilities',
                                          summary='No liabilities were selected.',
                                          details='No liabilities were selected from the Sequence liabilities ' +
                                                  'selection box in the data frame setup dialog.')]
            return DataFunctionResponse(outputColumns=[], notifications=notifications)

        color = string_input_field(request, 'ctrlColor')
        if color is None:
            color = '#FFFFFF'

        output_data, notifications = \
            self.run_sequence_liabilities(ab_sequences, liability_patterns, output_column_name, color)

        # build columns
        column_names = [name for name in output_data.keys()]
        columns = [ColumnData(name=output_column_name, dataType=DataType.BINARY, contentType='chemical/x-genbank',
                              values=[sequence_to_genbank_base64_str(s) for s in output_data[column_names[0]]])]
        for column_name in column_names[1:]:
            columns.append(ColumnData(name=column_name, dataType=DataType.BOOLEAN, values=output_data[column_name]))

        return DataFunctionResponse(outputColumns=columns, notifications=notifications)
