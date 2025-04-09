from collections import defaultdict
from typing import Any, Optional

from df.bio_helper import column_to_sequences, column_to_structures
from df.data_transfer import ColumnData, DataFunctionRequest, DataFunctionResponse, DataFunction, DataType, \
                             input_field_to_column
from df.data_transfer import Notification, NotificationLevel

from ruse.bio.antibody import Antibody

from Bio.SeqRecord import SeqRecord
from Bio.PDB.Structure import Structure


class AntibodyDevelopability(DataFunction):

    @staticmethod
    def run_AntibodyDevelopability(ab_sequences: list[Optional[SeqRecord]], ab_structures: list[Optional[Structure]]) \
            -> tuple[dict[str, dict[str, Any]], list[Notification]]:
        """
        Compute Antibody Developability Metrics

        :param ab_sequences:  A list of BioPython SeqRecords containing an Antibody sequence.
                           Expected to be consistent with the sequences in `structures`
        :param ab_structures:  list of BioPython Structure objects containing the protein structures of interest
        :return:  A tuple consisting of:
                     A dictionary with keys being strings with suggested column names and values of dictionaries.
                     The values for this dictionary is another dictionary having two possible keys:
                        'values' - containing the data values for the column
                        'properties' - containing property values to be applied to the column
        """

        notifications = []

        # sequences should:
        #   be a genbank column - this is implicit in the selection options
        #   have IMGT-Lefranc numbering
        #   have RASA annotations for all residues
        numbering_correct = False
        RASA_found = False

        for sequence in ab_sequences:
            for feature in sequence.features:
                if feature.type == 'misc_feature':
                    if not numbering_correct:
                        if 'cdr_definition: imgt_lefranc' in feature.qualifiers['note']:
                            numbering_correct = True
                    if not RASA_found:
                        if 'feature_name' in feature.qualifiers:
                            if 'Relative Accessible Surface Area' in feature.qualifiers['feature_name']:
                                RASA_found = True

                if numbering_correct and RASA_found:
                    break
            else:
                continue
            break

        if not numbering_correct:
            notifications.append(Notification(level=NotificationLevel.ERROR,
                                              title='Antibody Developability Metrics - CDR definition error',
                                              summary='CDR Definitions are not the correct format',
                                              details='For Antibody Developability Metrics, Antibody Numbering ' +
                                                      'must be IMGT (Lefranc).  This type of numbering is ' +
                                                      'automatically used during Antibody Structure Prediction.' +
                                                      'Verify the correct sequence column choice and Antibody ' +
                                                      'Numbering CDR Definitions are correct.'))

        if not RASA_found:
            notifications.append(Notification(level=NotificationLevel.ERROR,
                                              title='Antibody Developability Metrics - RASA not found.',
                                              summary='Relative Accessible Surface Area not found for the sequence',
                                              details='Relative Accessible Surface Area (RASA) must be calculated ' +
                                                      'for Antibody Developability Metrics calculations.'))

        if not numbering_correct or not RASA_found:
            return [], notifications

        CDR_lengths = defaultdict(list)
        PSHs = []
        PPCs = []
        PNCs = []
        SFvCSPs = []

        # notifications from metric function calls will be collected
        # in a dictionary and then formatted into a single notification
        # to prevent a large number of notifications that could occur row by row
        CDR_notifications = {}
        PSH_notifications = {}
        PPC_notifications = {}
        PNC_notifications = {}
        SFvCSP_notifications = {}

        for row, (sequence, structure) in enumerate(zip(ab_sequences, ab_structures)):
            if not sequence or not structure:
                for cdr in ['CDR-H1', 'CDR-H2', 'CDR-H3',
                            'CDR-L1', 'CDR-L2', 'CDR-L3']:
                    CDR_lengths[cdr].append(None)
                CDR_lengths['Total CDR'].append(None)
                PSHs.append(None)
                PPCs.append(None)
                PNCs.append(None)
                SFvCSPs.append(None)

                continue

            ab = Antibody(sequence, structure)

            (lengths, notifs) = ab.compute_CDR_lengths()
            if notifs:
                CDR_notifications[f'Row {row}'] = notifs

            total_cdr_length = 0
            for cdr in lengths:
                CDR_lengths[cdr].append(lengths[cdr])
                total_cdr_length += lengths[cdr]
            CDR_lengths['Total CDR'].append(total_cdr_length)

            (PSH, notifs) = ab.compute_PSH()
            PSHs.append(PSH)
            if notifs:
                PSH_notifications[f'Row {row}'] = notifs

            (PPC, notifs) = ab.compute_PPC()
            PPCs.append(PPC)
            if notifs:
                PPC_notifications[f'Row {row}'] = notifs

            (PNC, notifs) = ab.compute_PNC()
            PNCs.append(PNC)
            if notifs:
                PNC_notifications[f'Row {row}'] = notifs

            (SFvCSP, notifs) = ab.compute_SFvCSP()
            SFvCSPs.append(SFvCSP)
            if notifs:
                SFvCSP_notifications[f'Row {row}'] = notifs

        # compile notifications by row into single notification by metric
        if len(CDR_notifications) > 0:
            notifications.append(
                Notification.condense_notifications(CDR_notifications,
                                                    title = 'CDR Metric Calculation',
                                                    summary = 'Problems found during CDR Metric Calcualtion'))
        if len(PSH_notifications) > 0:
            notifications.append(
                Notification.condense_notifications(PSH_notifications,
                                                    title = 'PSH Metric Calculation',
                                                    summary='Problems found during PSH Metric Calcualtion'))
        if len(PPC_notifications) > 0:
            notifications.append(
                Notification.condense_notifications(PPC_notifications,
                                                    title = 'PPC Metric Calculation',
                                                    summary='Problems found during PPC Metric Calcualtion'))
        if len(PNC_notifications) > 0:
            notifications.append(
                Notification.condense_notifications(PNC_notifications,
                                                    title = 'PNC Metric Calculation',
                                                    summary='Problems found during PNC Metric Calcualtion'))
        if len(SFvCSP_notifications) > 0:
            notifications.append(
                Notification.condense_notifications(SFvCSP_notifications,
                                                    title = 'SFvCSP Metric Calculation',
                                                    summary='Problems found during SFvCSP Metric Calcualtion'))

        # count red and yellow alerts
        yellow_alerts = []
        red_alerts = []
        for index, (CDRtotal, PSH, PPC, PNC, SFvCSP) in (
                enumerate(zip(CDR_lengths['Total CDR'], PSHs, PPCs, PNCs, SFvCSPs))):

            if CDRtotal is None:
                red_alerts.append(None)
                yellow_alerts.append(None)
                continue

            yellow_alerts.append(0)
            red_alerts.append(0)

            if CDRtotal > 60:
                red_alerts[index] += 1
            elif CDRtotal > 53:
                yellow_alerts[index] += 1

            if PSH > 147.798 or PSH < 104.593:
                red_alerts[index] += 1
            elif PSH > 139.326 or PSH < 112.691:
                yellow_alerts[index] += 1

            if PPC > 2.927:
                red_alerts[index] += 1
            elif PPC >= 1.17:
                yellow_alerts[index] += 1

            if PNC > 2.95:
                red_alerts[index] += 1
            elif PNC >= 1.556:
                yellow_alerts[index] += 1

            if SFvCSP <= -15.83:
                red_alerts[index] += 1
            elif SFvCSP <= -4.55:
                yellow_alerts[index] += 1

        # build output columns
        output_column_data = {}
        for cdr, lengths in CDR_lengths.items():
            output_column_data[f'{cdr} Length'] = {'values': lengths}

        output_column_data['PSH'] = {'values': [round(value, 4) if value is not None else None for value in PSHs]}
        output_column_data['PPC'] = {'values': [round(value, 4) if value is not None else None for value in PPCs]}
        output_column_data['PNC'] = {'values': [round(value, 4) if value is not None else None for value in PNCs]}
        output_column_data['SFvCSP'] = {'values': [round(value, 4) if value is not None else None for value in SFvCSPs]}
        output_column_data['Yellow Alert Count'] = {'values': yellow_alerts}
        output_column_data['Red Alert Count'] = {'values': red_alerts}

        return output_column_data, notifications

    def execute(self, request: DataFunctionRequest) -> DataFunctionResponse:

        notifications = []

        sequence_column = input_field_to_column(request, 'uiSequenceColumn')
        ab_sequences = column_to_sequences(sequence_column)

        structure_column = input_field_to_column(request, 'uiStructureColumn')
        ab_structures, structure_notifications = column_to_structures(structure_column)
        notifications.extend(structure_notifications)

        output_column_data, devmet_notifications = self.run_AntibodyDevelopability(ab_sequences, ab_structures)
        notifications.extend(devmet_notifications)

        columns = []
        if len(output_column_data) != 0:
            for cdr in ['CDR-H1', 'CDR-H2', 'CDR-H3',
                        'CDR-L1', 'CDR-L2', 'CDR-L3', 'Total CDR']:
                columns.append(ColumnData(name=f'{cdr} Length', dataType=DataType.INTEGER,
                                          values=output_column_data[f'{cdr} Length']['values']))

            columns.append(ColumnData(name='Patches of Surface Hydrophobicity Metric (PSH)', dataType = DataType.FLOAT,
                                      values=output_column_data['PSH']['values']))
            columns.append(ColumnData(name='Patches of Positive Charge Metric (PPC)', dataType = DataType.FLOAT,
                                      values=output_column_data['PPC']['values']))
            columns.append(ColumnData(name='Patches of Negative Charge Metric (PNC)', dataType = DataType.FLOAT,
                                      values=output_column_data['PNC']['values']))
            columns.append(ColumnData(name='Structural Charge Symmetry Parameter (SFvCSP)', dataType = DataType.FLOAT,
                                      values=output_column_data['SFvCSP']['values']))
            columns.append(ColumnData(name='Developability Yellow Alert Count', dataType = DataType.INTEGER,
                                      values = output_column_data['Yellow Alert Count']['values']))
            columns.append(ColumnData(name='Developability Red Alert Count', dataType = DataType.INTEGER,
                                      values = output_column_data['Red Alert Count']['values']))

        return DataFunctionResponse(outputColumns=columns, notifications=notifications)
