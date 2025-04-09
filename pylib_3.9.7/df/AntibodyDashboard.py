from AntibodyStructurePrediction import AntibodyStructurePrediction as abStructPred
from ProteinRASA import ProteinRASA
from AntibodyDevelopability import AntibodyDevelopability as abDevMet
from AntibodySequenceLiabilities import AntibodySequenceLiabilities as abSeqLiab

from df.bio_helper import column_to_sequences, structures_to_column, column_to_structures
from df.data_transfer import ColumnData, TableData, DataFunctionRequest, DataFunctionResponse, DataFunction, DataType, \
    boolean_input_field, input_field_to_column, Notification, NotificationLevel

from ruse.bio.bio_data_table_helper import sequence_to_genbank_base64_str, genbank_base64_str_to_sequence


class AntibodyDashboard(DataFunction):
    """
    Creates an antibody dashboard by chaining Antibody Structure Prediction (which does Antibody Numbering),
    Relative Accessible Surface Area, Antibody Developability Metrics, and Antibody Sequence Liabilities
    """

    def execute(self, request: DataFunctionRequest) -> DataFunctionResponse:

        # Gather inputs
        sequence_column = input_field_to_column(request, 'uiAbSeqCol')
        sequence_column.remove_nulls()
        ab_sequences = column_to_sequences(sequence_column)

        pre_ab_ids = input_field_to_column(request, 'uiIDCol').values
        ab_ids = []
        for index, ab_id in enumerate(pre_ab_ids):
            if index in sequence_column.missing_null_positions:
                continue
            if ab_id is None or ab_id.strip() == '':
                ab_ids.append(f'Row {index + 1}')
            else:
                ab_ids.append(ab_id)

        notifications = []
        # run structure prediction
        struct_predict_column_data, struct_notifications = \
            abStructPred.run_AntibodyStructurePrediction(ab_sequences=ab_sequences, ab_ids=ab_ids, save_all=False)
        notifications.extend(struct_notifications)

        # prep needed data for RASA
        structure_column, embedding_notifications = \
            structures_to_column(struct_predict_column_data['PDB Structures']['values'])
        sequences = \
            [genbank_base64_str_to_sequence(gb64str, row_index=index)
             for index, gb64str in
             enumerate(struct_predict_column_data['Concatenated Chains (Heavy + Light)']['values'])]
        structures, struct_to_column_notifications = column_to_structures(structure_column)

        # run RASA and replace Antibody Numbering column with Antibody Numbering + RASA column
        rasa = ProteinRASA()
        rasa_column_data, rasa_notifications = \
            rasa.run_ProteinRASA(structures=structures, sequences=sequences,
                                 rasa_cutoff=7.5, exposed_color='#0D0DB8',
                                 label_all=True, start_color='#D67229', end_color='#0D0DB8')
        notifications.extend(rasa_notifications)

        # run developability metrics
        rasa_sequences = \
            [genbank_base64_str_to_sequence(gb64str, row_index=index)
             for index, gb64str in
             enumerate(rasa_column_data['Relative Accessible Surface Area Annotations']['values'])]
        metdev_column_data, metdev_notifications = abDevMet.run_AntibodyDevelopability(rasa_sequences, structures)
        notifications.extend(metdev_notifications)

        # run Sequence Liabilities with standard set
        seqliab_column_data, seqliab_notifications \
            = abSeqLiab.run_sequence_liabilities(ab_sequences=rasa_sequences,
                                                 patterns=abSeqLiab.SequenceLiabilityPatterns,
                                                 output_column_name='Annotated Liabilities',
                                                 color='#FF0000')
        notifications.extend(seqliab_notifications)

        # compile columns and build table
        columns = [ColumnData(name='ID', dataType=DataType.STRING, values=struct_predict_column_data['ID']['values'])]

        # fully elaborated GenBank column with all annotations
        column_names = [name for name in seqliab_column_data.keys()]
        columns.append(
            ColumnData(name='Annotated Antibody Sequence', dataType=DataType.BINARY, contentType='chemical/x-genbank',
                       values=[sequence_to_genbank_base64_str(s) for s in seqliab_column_data[column_names[0]]]))

        # Sequence Liabilities
        for column_name in column_names[1:]:
            columns.append(ColumnData(name=column_name, dataType=DataType.BOOLEAN,
                                      values=seqliab_column_data[column_name]))

        # Developability Metrics
        columns.extend([ColumnData(name='Developability Yellow Alert Count', dataType=DataType.INTEGER,
                                   values=metdev_column_data['Yellow Alert Count']['values']),
                        ColumnData(name='Developability Red Alert Count', dataType=DataType.INTEGER,
                                   values=metdev_column_data['Red Alert Count']['values']),
                        ColumnData(name='Patches of Surface Hydrophobicity Metric (PSH)', dataType=DataType.FLOAT,
                                   properties={'Alias': 'PSH'},
                                   values=metdev_column_data['PSH']['values']),
                        ColumnData(name='Patches of Positive Charge Metric (PPC)', dataType=DataType.FLOAT,
                                   properties={'Alias': 'PPC'},
                                   values=metdev_column_data['PPC']['values']),
                        ColumnData(name='Patches of Negative Charge Metric (PNC)', dataType=DataType.FLOAT,
                                   properties={'Alias': 'PNC'},
                                   values=metdev_column_data['PNC']['values']),
                        ColumnData(name='Structural Charge Symmetry Parameter (SFvCSP)', dataType=DataType.FLOAT,
                                   properties={'Alias': 'SFvCSP'},
                                   values=metdev_column_data['SFvCSP']['values']),
                        ColumnData(name='Total CDR Length', dataType=DataType.INTEGER,
                                   properties={'Alias': 'Total CDR Length'},
                                   values=metdev_column_data['Total CDR Length']['values'])])
        for cdr in ['CDR-H1', 'CDR-H2', 'CDR-H3',
                    'CDR-L1', 'CDR-L2', 'CDR-L3']:
            columns.append(ColumnData(name=f'{cdr} Length', dataType=DataType.INTEGER,
                                      values=metdev_column_data[f'{cdr} Length']['values']))

        # Predicted Structure and sequences
        columns.extend([structure_column,
                        ColumnData(name='Original Sequence', dataType=DataType.STRING,
                                   contentType='chemical/x-sequence',
                                   values=struct_predict_column_data['Original Sequence']['values']),
                        ColumnData(name='Heavy Chain', dataType=DataType.STRING,
                                   contentType='chemical/x-sequence',
                                   values=struct_predict_column_data['Heavy Chain']['values']),
                        ColumnData(name='Light Chain', dataType=DataType.STRING,
                                   contentType='chemical/x-sequence',
                                   values=struct_predict_column_data['Light Chain']['values'])])

        output_table = TableData(tableName='Antibody Dashboard Data', columns=columns)

        return DataFunctionResponse(outputTables=[output_table], notifications=notifications)
