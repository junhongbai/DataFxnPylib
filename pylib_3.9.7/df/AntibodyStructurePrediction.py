import os
import traceback
from typing import Any, Optional, Union

from df.bio_helper import column_to_sequences, structures_to_column
from df.data_transfer import ColumnData, TableData, DataFunctionRequest, DataFunctionResponse, DataFunction, DataType, \
    boolean_input_field, input_field_to_column, Notification, NotificationLevel

from ruse.bio.antibody import align_antibody_sequences, NumberingScheme, CDRDefinitionScheme, \
    AntibodySequencePair, extract_ABB2_numbering

from ImmuneBuilder import ABodyBuilder2
from ImmuneBuilder.ABodyBuilder2 import header as ABB2_HEADER
from ImmuneBuilder.refine import refine
from ImmuneBuilder.util import add_errors_as_bfactors

from Bio import SeqRecord


class AntibodyStructurePrediction(DataFunction):
    """
    Predicts antibody structure from heavy and light chain sequences
    """

    @staticmethod
    def run_AntibodyStructurePrediction(ab_sequences: list[SeqRecord],
                                        ab_ids: list[Union[str, int]], save_all: bool = False) -> \
            tuple[dict[str, Any], list[Optional[Notification]]]:
        """
        Predict antibody structure from sequence, and return data for output columns

        :param ab_sequences:  A list of BioPython SeqRecords containing an Antibody sequence.
                              Expected to be Heavy Chain + Light Chain concatenated
        :param ab_ids: A list of values to use for antibody IDs.  Typically either strings or integers.
        :param save_all:  Boolean - if true, results for all 4 models will be saved,
                                    if false, only one model result will be saved
        :return:  A tuple consisting of:
                     A dictionary with keys being strings with suggested column names and values of dictionaries.
                     The values for this dictionary is another dictionary having two possible keys:
                        'values' - containing the data values for the column
                        'properties' - containing property values to be applied to the column
        """

        if save_all:
            row_multiplier = 4
        else:
            row_multiplier = 1

        # create an ABodyBuilder2 structure predictor with the selected numbering scheme
        predictor = ABodyBuilder2(numbering_scheme='imgt')

        # new table columns
        ids = []

        model_number = []
        model_rank = []

        orig_seq = []
        HL_concat_seq = []
        heavy_chain_seq = []
        light_chain_seq = []
        pdb_strings = []
        ABB2_numbering = []

        notifications = []
        structure_prediction_notifications = {}
        structure_saving_notifications = {}

        for ab_seq, ab_id in zip(ab_sequences, ab_ids):
            ids.extend([ab_id] * row_multiplier)
            orig_seq.extend([str(ab_seq.seq)] * row_multiplier)

            # concatenated sequence is provided for heavy and light chains
            # ABB algorithm identifies individual chains
            sequences = {'H': str(ab_seq.seq).upper(), 'L': str(ab_seq.seq).upper()}
            try:
                antibody = predictor.predict(sequences)
            except Exception as ex:
                structure_prediction_notifications[ab_id] = (ex, f'{traceback.format_exc()}')

                # remove list elements for this broken item
                ids = ids[:-row_multiplier]
                orig_seq = orig_seq[:-row_multiplier]

                continue

            heavy_chain_seq.extend(
                [''.join(residue[1] for residue in antibody.numbered_sequences['H'])] * row_multiplier)
            light_chain_seq.extend(
                [''.join(residue[1] for residue in antibody.numbered_sequences['L'])] * row_multiplier)
            HL_concat_seq.extend([''.join([heavy_chain_seq[-1], light_chain_seq[-1]])] * row_multiplier)
            ABB2_numbering.extend([antibody.numbered_sequences] * row_multiplier)

            if save_all:
                model_number.extend([idx + 1 for idx in range(row_multiplier)])
                model_rank.extend([idx + 1 for idx in antibody.ranking])

                for model_idx in range(len(antibody.atoms)):
                    pdb_filename = ''.join([ab_id, f'_Model_{model_idx}_Rank_{antibody.ranking.index(model_idx)}.pdb'])
                    error_caught = False
                    try:
                        antibody.save_single_unrefined(pdb_filename, index=model_idx)
                        refine(pdb_filename, pdb_filename, check_for_strained_bonds=True, n_threads=-1)
                        add_errors_as_bfactors(pdb_filename, antibody.error_estimates.mean(0).sqrt().cpu().numpy(),
                                               header=[ABB2_HEADER])

                        # # open the file, compress, and encode it
                        with open(pdb_filename, 'r', encoding='utf-8') as pdb_file:
                            pdb_data = pdb_file.read()
                            pdb_strings.append(pdb_data)
                    except Exception as ex:
                        structure_prediction_notifications[ab_id] = (ex, f'{traceback.format_exc()}')

                        # remove list elements for this broken item
                        ids = ids[:-row_multiplier]
                        orig_seq = orig_seq[:-row_multiplier]
                        heavy_chain_seq = heavy_chain_seq[:-row_multiplier]
                        light_chain_seq = light_chain_seq[:-row_multiplier]
                        HL_concat_seq = HL_concat_seq[:-row_multiplier]
                        model_rank = model_rank[:-row_multiplier]
                        model_number = model_number[:-row_multiplier]

                        error_caught = True
                    finally:
                        if os.path.exists(pdb_filename):
                            os.remove(pdb_filename)
                        if error_caught:
                            continue
            else:
                pdb_filename = '_'.join([ab_id, 'predicted.pdb'])
                error_caught = False
                try:
                    antibody.save_single_unrefined(pdb_filename)
                    refine(pdb_filename, pdb_filename, check_for_strained_bonds=True, n_threads=-1)

                    # # open the file, compress, and encode it
                    with open(pdb_filename, 'r', encoding='utf-8') as pdb_file:
                        pdb_data = pdb_file.read()
                        pdb_strings.append(pdb_data)
                except Exception as ex:
                    structure_saving_notifications[ab_id] = (ex, f'{traceback.format_exc()}')

                    # remove list elements for this broken item
                    ids = ids[:-row_multiplier]
                    orig_seq = orig_seq[:-row_multiplier]
                    heavy_chain_seq = heavy_chain_seq[:-row_multiplier]
                    light_chain_seq = light_chain_seq[:-row_multiplier]
                    HL_concat_seq = HL_concat_seq[:-row_multiplier]
                    model_rank = model_rank[:-row_multiplier]
                    model_number = model_number[:-row_multiplier]

                    error_caught = True
                finally:
                    if os.path.exists(pdb_filename):
                        os.remove(pdb_filename)
                    if error_caught:
                        continue

        # antibody numbering
        HL_chain_values = HL_concat_seq
        HL_chain_props = {}

        try:
            # get alignment and numbering information from ABB2 results
            HL_chain_values, HL_chain_props = \
                extract_ABB2_numbering(ABB2_numbering,
                                       [AntibodySequencePair(H=h_seq, L=l_seq)
                                        for h_seq, l_seq in zip(heavy_chain_seq, light_chain_seq)])

        except Exception as ex:
            notifications.append(Notification(level=NotificationLevel.ERROR,
                                              title='Antibody Structure Prediction - Antibody Numbering',
                                              summary=f'An unexpected error occurred\n{ex.__class__} - {ex}',
                                              details=f'{traceback.format_exc()}'))

        # compile notifications
        if len(structure_prediction_notifications) != 0:
            notifications.append(Notification(level=NotificationLevel.ERROR,
                                              title='Antibody Structure Prediction',
                                              summary=f'Error for ID(s) ' +
                                                      '/n'.join(['/t' + str(ab_id)
                                                                for ab_id in
                                                                structure_prediction_notifications.keys()]) +
                                                      '.\n',
                                              details=''.join([f'{ab_id}\n{ex.__class__} - {ex}\n' + f'{details}\n\n'
                                                               for ab_id, (ex, details) in
                                                               structure_prediction_notifications.items()])))

        if len(structure_saving_notifications) != 0:
            notifications.append(Notification(level=NotificationLevel.ERROR,
                                              title='Antibody Structure Prediction - File error',
                                              summary=f'Error for ID(s) ' +
                                                      '/n'.join(['/t' + str(ab_id)
                                                                for ab_id in
                                                                structure_saving_notifications.keys()]) +
                                                      '.\n',
                                              details=''.join([f'{ab_id}\n{ex.__class__} - {ex}\n' + f'{details}\n\n'
                                                               for ab_id, (ex, details) in
                                                               structure_saving_notifications.items()])))

        # gather output column data
        output_column_data = {'ID': {'values': ids}}
        if save_all:
            output_column_data.update({'Model Number': {'values': model_number},
                                       'Model Rank': {'values': model_rank}})
        output_column_data.update({'PDB Structures': {'values': pdb_strings},
                                   'Concatenated Chains (Heavy + Light)': {'values': HL_chain_values,
                                                                           'properties': HL_chain_props},
                                   'Original Sequence': {'values': orig_seq},
                                   'Heavy Chain': {'values': heavy_chain_seq},
                                   'Light Chain': {'values': light_chain_seq}
                                   })

        return output_column_data, notifications

    def execute(self, request: DataFunctionRequest) -> DataFunctionResponse:

        sequence_column = input_field_to_column(request, 'uiAbSeqCol')
        sequence_column.remove_nulls()
        ab_sequences = column_to_sequences(sequence_column)

        id_column = input_field_to_column(request, 'uiIDCol')
        if 'contentType' in id_column and 'chemical' in id_column.contentType:
            notifications = [Notification(level=NotificationLevel.ERROR,
                                              title='Antibody Structure Prediction',
                                              summary='ID column - wrong data type',
                                              details='The ID column should not be of any type other than a string ' +
                                                      'or a numeric data type. ' +
                                                      f'Selected column was of type {id_column.contentType}')]

            return DataFunctionResponse(outputTables=[], notifications=notifications)

        pre_ab_ids = id_column.values
        ab_ids = []
        for index, ab_id in enumerate(pre_ab_ids):
            if index in sequence_column.missing_null_positions:
                continue
            if ab_id is None or ab_id.strip() == '':
                ab_ids.append(f'Row {index + 1}')
            else:
                ab_ids.append(ab_id)

        save_all = boolean_input_field(request, 'uiSaveAll')

        # output_table, notifications = self.run_AntibodyStructurePrediction(ab_sequences, ab_ids, save_all)
        output_column_data, notifications = self.run_AntibodyStructurePrediction(ab_sequences, ab_ids, save_all)

        # convert predicted structures to proper ColumnData object
        structure_column, embedding_notifications = \
            structures_to_column(output_column_data['PDB Structures']['values'])
        notifications.extend(embedding_notifications)

        columns = [ColumnData(name='ID', dataType=DataType.STRING, values=output_column_data['ID']['values']),
                   structure_column,
                   ColumnData(name='Concatenated Chains (Heavy + Light)', dataType=DataType.BINARY,
                              contentType='chemical/x-genbank',
                              values=output_column_data['Concatenated Chains (Heavy + Light)']['values'],
                              properties=output_column_data['Concatenated Chains (Heavy + Light)']['properties']),
                   ColumnData(name='Original Sequence', dataType=DataType.STRING,
                              contentType='chemical/x-sequence',
                              values=output_column_data['Original Sequence']['values']),
                   ColumnData(name='Heavy Chain', dataType=DataType.STRING,
                              contentType='chemical/x-sequence',
                              values=output_column_data['Heavy Chain']['values']),
                   ColumnData(name='Light Chain', dataType=DataType.STRING,
                              contentType='chemical/x-sequence',
                              values=output_column_data['Light Chain']['values'])]

        if save_all:
            columns[1:1] = [ColumnData(name='Model Number', dataType=DataType.INTEGER,
                                       values=output_column_data['Model Number']['values']),
                            ColumnData(name='Model Rank', dataType=DataType.INTEGER,
                                       values=output_column_data['Model Rank']['values'])]

        # compile output table
        output_table = TableData(tableName='Antibody Structure Predictions',
                                 columns=columns)

        return DataFunctionResponse(outputTables=[output_table], notifications=notifications)
