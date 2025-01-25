import os
import csv
import pandas as pd
import pm4py
import numpy as np

from timeit import default_timer as timer
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.conformance.alignments.petri_net import algorithm
import pm4py.algo.conformance.alignments.petri_net.variants.incremental_a_star as incas
import pm4py.algo.conformance.alignments.petri_net.variants.incremental_a_star_mp as mocc
from pm4py.objects.petri_net.utils.align_utils import SKIP

# ------------------------------------- [ Configure Inputs and Options ] -----------------------------------------------

processes = {
    "al": {
        "input_model_name": "al_dpn_model.pnml",
        "input_file_names": ["al_good_traces.csv", "al_bad_traces.csv", "al_one_day_prod.csv"],
        "str_attributes": ["activity", "rotary_table"]
    },
    "fm": {
        "input_model_name": "fm_dpn_model.pnml",
        "input_file_names": ["fm_good_traces.csv", "fm_bad_traces.csv"],
        "str_attributes": ["case", "event", "dismissal", "notificationType", "org:resource", "vehicleClass"]
    },
    "hb": {
        "input_model_name": "hb_dpn_model.pnml",
        "input_file_names": ["hb_good_traces.csv", "hb_bad_traces.csv"],
        "str_attributes": ["speciality", "closeCode", "caseType"]
    }
}

# Options for calculating the alignments
process = "al"                                  # Assembly Line Process ("al"), Road Traffic Fine Management Process ("fm"), Hospital Billing Process ("hb")
algo_type = "mp"                                # control-flow ("cf"), multi-perspective ("mp")
mode = 0                                        # online (0), offline (1)
input_file_index = 0                            # good traces (0), bad traces (1), one day production (2)
input_path = os.path.join("input")              # input folder with the model file and event log file
ova_output_path = os.path.join("ova_outputs")   # output folder for the OVA calculations
alignment_output_file = "alignment_outputs.txt" # file name for the calculated alignments
time_output_file = "time_outputs.txt"           # file name for the measured calculation times of each online alignment

# Options for saving the calculated alignments as alignment data (used for visualization)
output_type = 0                                 # event stream type: composite (0), single (1)
alignment_output_file_vis = "alignment_data.js" # file name for the event data

config = processes[process]
input_model_name = config["input_model_name"]
input_file_name = config["input_file_names"][input_file_index]
str_attributes = config["str_attributes"]

# --------------------------------------- [ Calculate the Alignments ] ------------------------------------------------

# Read CSV file
def read_csv_file(csv_file_path):
    global attributes
    
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        attributes = next(reader)[0].split(";")
    
    df = pd.read_csv(csv_file_path, sep=';', quotechar="\"", parse_dates=[2])
    if process == "al":
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        df['complete_timestamp'] = pd.to_datetime(df['complete_timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case_id"}
    elif process == "fm":
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case"}
    elif process == "hb":
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
        parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case"}        
    # EventLog -> Trace, Event
    return log_converter.apply(df, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

# Compute alignments
def apply_incremental_a_star(event_log, net, initial_marking, final_marking, algo_type, mode):
    parameters = {}
    if process == "al":
        parameters[algorithm.VERSION_STATE_EQUATION_A_STAR.value.Parameters.CASE_ID_KEY] = "case_id"
        parameters[algorithm.VERSION_STATE_EQUATION_A_STAR.value.Parameters.ACTIVITY_KEY] = "activity"
    elif process == "fm" or process == "hb":
        parameters[algorithm.VERSION_STATE_EQUATION_A_STAR.value.Parameters.CASE_ID_KEY] = "case"
        parameters[algorithm.VERSION_STATE_EQUATION_A_STAR.value.Parameters.ACTIVITY_KEY] = "event"      
        
    alignment_list = []
    
    if algo_type == "cf":
        for trace in event_log:
            trace_alignment = incas.apply(trace, net, initial_marking, final_marking, parameters=parameters, mode=mode)
            alignment_list.append(trace_alignment)
    
    if algo_type == "mp":
        for trace in event_log:
            trace_alignment = mocc.apply(ova_output_path, trace, net, initial_marking, final_marking, parameters=parameters, mode=mode)
            alignment_list.append(trace_alignment)
    
    return alignment_list


attributes = None

# Read PNML file (process model)
net, initial_marking, final_marking = pnml_importer.apply(os.path.join(input_path, input_model_name), parameters={"type":"DPN"})

# Read CSV file (event log)
log = read_csv_file(os.path.join(input_path, input_file_name))

# Compute alignments
alignment_list = apply_incremental_a_star(log, net, initial_marking, final_marking, algo_type, mode)

# ------------------------------------------ [ Process the Output ] ------------------------------------------------------

"""
move_type:
1	corr. sync.
2	incorr. sync.
3	log
4	model (if observable activity)
use corrected_values only if move_type=2
to connect: y_attribue, activity, start_timestamp, end_timestamp
"""

last_attr_values = dict()
for attr in attributes:
    last_attr_values[attr] = np.nan

a_steps_list = []
for alignment in alignment_list:
    for ali in alignment['alignment']:
        if ali['label'][1] != None:     # if it has an activity
            # udapte variable assignments
            if ali['label'][0] != SKIP:
                if 'attribute_values' in ali:
                    for key, value in ali['attribute_values'].items():
                        last_attr_values[key] = value        
            a_step_dict = dict()
            a_step_dict["values"] = dict()
            a_step_dict["values"]["case_id"] = alignment['case-id']
            # sync. move
            if ali['label'][0] == ali['label'][1]:
                a_step_dict["values"]["activity"] = ali['label'][0]
                # correct sync. move
                if 'deviations' not in ali:
                    a_step_dict["move_type"] = 1
                # incorrect sync. move
                else:
                    a_step_dict["move_type"] = 2
                    a_step_dict["corrected_values"] = dict()
                    for dev in ali['deviations']:
                        a_step_dict["corrected_values"][dev[0]] = ali['variable_assignments'][dev[0]]
            else:
                # log move   
                if ali['label'][0] != SKIP:
                    a_step_dict["values"]["activity"] = ali['label'][0]
                    a_step_dict["move_type"] = 3
                # model move
                else:
                    a_step_dict["values"]["activity"] = ali['label'][1]
                    a_step_dict["move_type"] = 4
            # get values
            if a_step_dict["move_type"] != 4:
                if 'attribute_values' in ali:
                    for key, value in ali['attribute_values'].items():
                        a_step_dict["values"][key] = value
            else:
                if 'variable_assignments' in ali:
                    for key, value in ali['variable_assignments'].items():
                         a_step_dict["values"][key] = value
                for key, value in last_attr_values.items():
                    if key not in a_step_dict["values"]:
                        a_step_dict["values"][key] = value
                if process == "al":
                    a_step_dict["values"]["start_timestamp"] = a_step_dict["values"]["complete_timestamp"]
                elif process == "fm":
                    a_step_dict["values"]["start_timestamp"] = a_step_dict["values"]["date"]
                elif process == "hb":
                    a_step_dict["values"]["start_timestamp"] = a_step_dict["values"]["datetime"]
            
            a_steps_list.append(a_step_dict)

# Add attribute values to model moves if they don't have them yet (it can happen if the first x alignment moves are model moves)
for a_step_dict in reversed(a_steps_list):
    for key, value in a_step_dict["values"].items():
        last_attr_values[key] = value
    if a_step_dict["move_type"] == 4:
        for key, value in last_attr_values.items():
            if key not in a_step_dict["values"]:
                a_step_dict["values"][key] = value
        if process == "al":
            if a_step_dict["values"]["complete_timestamp"] != a_step_dict["values"]["start_timestamp"]:
                a_step_dict["values"]["complete_timestamp"] = a_step_dict["values"]["start_timestamp"]

# Print the cost of the calculated alignments
for alignment in alignment_list:
    print(alignment['case-id'], alignment['cost'])

# ---------------------------------------- [ Save the Output ] ------------------------------------------------------

# Save the calculated alignments into a file
with open(alignment_output_file, 'w') as f:
    for alignment in alignment_list:
        f.write(str(alignment['case-id'])+'\n')
        f.write(str(alignment['cost'])+'\n')
        for ali in alignment['alignment']:
            f.write('\n')
            if ali['label'][1] != None:
                f.write(str(ali['label'])+'\n')
            else:
                f.write(str(ali['name'])+'\n')
            if 'attribute_values' in ali:
                f.write('attribute_values\t')
                f.write(str(ali['attribute_values'])+'\n')
            if 'variable_assignments' in ali:
                f.write('variable_assignments\t')
                f.write(str(ali['variable_assignments'])+'\n')
            if 'deviations' in ali:
                f.write('deviations\t')
                f.write(str(ali['deviations'])+'\n')
        f.write('\n')
        f.write('\n')

# Save the computation time of each online alignment
with open(time_output_file, 'w') as f:
    case_id = 0
    for alignment in alignment_list:
        case_id = alignment["case-id"]
        for res in alignment["intermediate_results"]:
            f.write(str(case_id) + "\t" + str(res["event_computation_time"]) + "\n")

# Composite (start & complete) event per observable unit
if output_type == 0:
    with open(alignment_output_file_vis, 'w') as f:
        f.write("var alignment_data = [\n")
        a_id = 1
        for idx, a_step_dict in enumerate(a_steps_list):
            str_a = "{ id: " + str(a_id) + ", move_type: " + str(a_step_dict["move_type"]) + ", values: {"
            values = []
            for key, value in a_step_dict["values"].items():
                value_str = str(value)
                if value_str != "nan":
                    if (key == "start_timestamp") or (key == "complete_timestamp") or (key == "date"):
                        values.append(key + ": new Date(\"" + value_str + "\")")
                    elif key in str_attributes:
                        values.append(key + ": '" + value_str + "'")
                    else:
                        values.append(key + ": " + value_str)
            str_a += ", ".join(values) + "}"
            if a_step_dict["move_type"] == 2:
                str_a += ", corrected_values: {"
                str_a += ", ".join(f"{key}: {value}" for key, value in a_step_dict["corrected_values"].items()) + "}"
            if idx == len(a_steps_list) - 1:  # Check if it is the last iteration
                str_a += " }]"
            else:
                str_a += " },\n"
            f.write(str_a)
            a_id += 1

# Single event per observable unit
if output_type == 1:
    with open(alignment_output_file_vis, 'w') as f:
        f.write("var alignment_data = [\n")
        a_id = 1
        ai_id = 1
        for idx, a_step_dict in enumerate(a_steps_list):
            for tr_type in ["start", "complete"]:
                str_a = "{ id: " + str(a_id) + ", move_type: " + str(a_step_dict["move_type"]) + ", values: {"
                values = []
                values.append("ai_id: " + str(ai_id))
                values.append("tr_type: '" + tr_type + "'")
                for key, value in a_step_dict["values"].items():
                    value_str = str(value)
                    if value_str != "nan":
                        if (key == "start_timestamp" and tr_type == "start") or (key == "complete_timestamp" and tr_type == "complete"):
                            values.append("timestamp: new Date(\"" + value_str + "\")")
                        elif (key == "start_timestamp" and tr_type == "complete") or (key == "complete_timestamp" and tr_type == "start"):
                            continue
                        elif key in str_attributes:
                            values.append(key + ": '" + value_str + "'")
                        else:
                            values.append(key + ": " + value_str)
                str_a += ", ".join(values) + "}"
                if a_step_dict["move_type"] == 2:
                    str_a += ", corrected_values: {"
                    str_a += ", ".join(f"{key}: {value}" for key, value in a_step_dict["corrected_values"].items()) + "}"
                if idx == len(a_steps_list) - 1:  # Check if it is the last iteration
                    str_a += " }]"
                else:
                    str_a += " },\n"
                f.write(str_a)
                a_id += 1
            ai_id += 1
