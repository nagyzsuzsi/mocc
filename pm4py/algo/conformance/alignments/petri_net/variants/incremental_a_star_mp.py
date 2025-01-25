'''
    This file is NOT part of PM4Py.
    Author: Zsuzsanna Nagy
'''

import heapq
import time
import math

from ortools.linear_solver import pywraplp

from pm4py.algo.conformance.alignments import petri_net

from pm4py.objects import petri_net as petri
from pm4py.objects.log.obj import Trace
from pm4py.algo.conformance.alignments.petri_net.utils import mp_utils as mu

from enum import Enum
from pm4py.util import exec_utils
from pm4py.objects.log import obj as log_implementation
from pm4py.objects.petri_net.utils import align_utils as utils
from pm4py.objects.petri_net.utils import petri_utils as p_utils
from pm4py.objects.petri_net.utils import synchronous_product
from pm4py.objects.petri_net.utils.align_utils import SKIP
from pm4py.util.xes_constants import DEFAULT_NAME_KEY, DEFAULT_TRACEID_KEY
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY, PARAMETER_CONSTANT_CASEID_KEY


model_move_probabilities_for_heuristic = None

# key: (cf_alignment_process_part, variable_writings), value: {ovas, g_var, current_variable_values}
ova_cache = dict()


class Parameters(Enum):
    PARAM_TRACE_COST_FUNCTION = 'trace_cost_function'
    PARAM_MODEL_COST_FUNCTION = 'model_cost_function'
    PARAM_SYNC_COST_FUNCTION = 'sync_cost_function'
    PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE = 'ret_tuple_as_trans_desc'
    PARAM_TRACE_NET_COSTS = "trace_net_costs"
    TRACE_NET_CONSTR_FUNCTION = "trace_net_constr_function"
    TRACE_NET_COST_AWARE_CONSTR_FUNCTION = "trace_net_cost_aware_constr_function"
    PARAM_MAX_ALIGN_TIME_TRACE = "max_align_time_trace"
    PARAM_MAX_ALIGN_TIME = "max_align_time"
    PARAMETER_VARIANT_DELIMITER = "variant_delimiter"
    CASE_ID_KEY = PARAMETER_CONSTANT_CASEID_KEY
    ACTIVITY_KEY = PARAMETER_CONSTANT_ACTIVITY_KEY
    VARIANTS_IDX = "variants_idx"


PARAM_TRACE_COST_FUNCTION = Parameters.PARAM_TRACE_COST_FUNCTION.value
PARAM_MODEL_COST_FUNCTION = Parameters.PARAM_MODEL_COST_FUNCTION.value
PARAM_SYNC_COST_FUNCTION = Parameters.PARAM_SYNC_COST_FUNCTION.value

output_path = None

def apply(op, trace, petri_net, initial_marking, final_marking, parameters=None, debug_print=False, derive_heuristic=True,
          dijkstra=False, recalculate_heuristic_open_set=True, alignment_cost=None, variable_cost=None, mode=0):
    # possible modes: online (0), offline (1)
    
    global output_path
    
    start_time = time.time()
    duration_solving_lps_total = 0
    output_path = op

    case_key = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, DEFAULT_TRACEID_KEY)
    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, DEFAULT_NAME_KEY)
    
    variable_cost = dict() if variable_cost == None else variable_cost
    if len(variable_cost) == 0:
        for var in petri_net.variables:
            variable_cost[var.name] = (1,1)
            #variable_cost[var.name] = (0.1,0.1)
    
    def_alignment_cost = dict() if alignment_cost == None else alignment_cost
    if alignment_cost == None:
        alignment_cost = dict()   
        alignment_cost["log"] = 1
        alignment_cost["model"] = 1
        alignment_cost["inc_syn"] = 0
        alignment_cost["cor_syn"] = 0
    
    incremental_trace = Trace()
    print(incremental_trace)
    
    # create empty closed and open set
    open_set = []
    closed_set = set()
    first_event = True
    alignment = None

    visited_states_total = 0
    traversed_arcs_total = 0
    queued_states_total = 0
    intermediate_results = []
    heuristic_computation_time_total = 0
    number_solved_lps_total = 0

    print('\n'+str(trace[0][case_key]))
    for event in trace:
        last_event = True if mode == 1 and event == trace[-1] else False
        print("\n event: ", event)
        start_time_trace = time.time()
        prev_time_event = time.time()
        incremental_trace.append(event)
        print("incremental_trace")
        print(incremental_trace)
        print(str(incremental_trace[-1][activity_key]))
        if debug_print:
            print(incremental_trace)
        if first_event:
            # activity_key: :class:`str` key of the attribute of the events that defines the activity name
            trace_net, trace_im, trace_fm = p_utils.construct_trace_net(incremental_trace,
                                                                            activity_key=activity_key)
            sync_prod, sync_im, sync_fm = synchronous_product.construct(trace_net,
                                                                              trace_im,
                                                                              trace_fm,
                                                                              petri_net,
                                                                              initial_marking,
                                                                              final_marking,
                                                                              SKIP)
            first_event = False
        else:
            sync_prod, sync_fm = synchronous_product.extend_trace_net_of_synchronous_product_net(sync_prod, event,
                                                                                                       sync_fm, SKIP,
                                                                                                       activity_key)


        cost_function = utils.construct_standard_cost_function(sync_prod, SKIP)
        
        prefix_alignment, open_set, closed_set, duration_solving_lps, number_solved_lps = __search(petri_net,
                                                                                                   sync_prod, sync_im,
                                                                                                   sync_fm,
                                                                                                   cost_function,
                                                                                                   SKIP, open_set,
                                                                                                   closed_set,
                                                                                                   incremental_trace,
                                                                                                   case_key,
                                                                                                   activity_key,
                                                                                                   derive_heuristic=derive_heuristic,
                                                                                                   dijkstra=dijkstra,
                                                                                                   recalculate_heuristic_open_set=recalculate_heuristic_open_set,
                                                                                                   alignment_cost=def_alignment_cost,
                                                                                                   variable_cost=variable_cost, complete_trace=last_event)
               
        # print(milp.print_model())
        # print(milp.print_solution())
        
        duration_solving_lps_total += duration_solving_lps
        alignment = prefix_alignment

        # update statistic values
        visited_states_total += prefix_alignment['visited_states']
        traversed_arcs_total += prefix_alignment['traversed_arcs']
        queued_states_total += prefix_alignment['queued_states']
        heuristic_computation_time_total += duration_solving_lps
        number_solved_lps_total += number_solved_lps

        res = {'trace_length': len(incremental_trace),
               'alignment': prefix_alignment['alignment'],
               'cost': prefix_alignment['cost'],
               'visited_states': prefix_alignment['visited_states'],
               'queued_states': prefix_alignment['queued_states'],
               'traversed_arcs': prefix_alignment['traversed_arcs'],
               'event_computation_time': time.time() - prev_time_event,
               'total_computation_time': time.time() - start_time_trace,
               'heuristic_computation_time': duration_solving_lps,
               'number_solved_lps': number_solved_lps}
        intermediate_results.append(res)
        prev_time_event = time.time() - prev_time_event
        if debug_print:
            print(prefix_alignment)
            print("cost: ", prefix_alignment["cost"])
            print(open_set)
            print(closed_set)
            print("\n\n---------------------------------------------------\n\n")

    duration_total = time.time() - start_time
    res = {'case-id': event[case_key],
           'alignment': alignment['alignment'],
           'cost': alignment['cost'],
           'visited_states': visited_states_total, 'queued_states': queued_states_total,
           'traversed_arcs': traversed_arcs_total, 'total_computation_time': duration_total,
           'heuristic_computation_time': heuristic_computation_time_total,
           'number_solved_lps': number_solved_lps_total, 'intermediate_results': intermediate_results}
    
    print(event[case_key], alignment['cost'], duration_total)
    
    return res

def __get_attribute_values(event, case_key, activity_key):
    attribute_values = dict(event._dict)
    del attribute_values[case_key]
    del attribute_values[activity_key]
    return attribute_values

def __search(petri_net, sync_net, initial_m, final_m, cost_function, skip, open_set_heap, closed_set, 
             incremental_trace, case_key, activity_key, 
             derive_heuristic=False, dijkstra=False, recalculate_heuristic_open_set=True,
             alignment_cost=None, variable_cost=None, complete_trace=False):
    number_solved_lps = 0
    duration_solving_lps = 0
    # initialize undiscovered states
    if len(open_set_heap) == 0:
        if dijkstra:
            h, x, duration = 0, None, 0
        else:
            h, x, duration = __compute_heuristic_regular_cost(sync_net, initial_m, final_m, cost_function)
            number_solved_lps += 1
        duration_solving_lps += duration
        ini_state = SearchTuple(0 + h, (0,0), h, initial_m, None, None, x, True)
        for var in petri_net.variables:
            ini_state.current_variable_values[var.name] = var.init_val       
        open_set_heap = [ini_state]
    # recalculate heuristic and update f-values
    else:
        if recalculate_heuristic_open_set:
            # recalculate heuristic for all markings in open set
            for st in open_set_heap:
                if dijkstra:
                    h, x, duration = 0, None, 0
                else:
                    h, x, duration = __compute_heuristic_regular_cost(sync_net, st.m, final_m, cost_function)
                    number_solved_lps += 1               
                duration_solving_lps += duration

                st.h = h
                st.f = st.g[0]+st.g[1] + st.h
                st.x = x
            heapq.heapify(open_set_heap)  # visited markings
        else:
            for st in open_set_heap:
                st.trust = False
            heapq.heapify(open_set_heap)
    visited = 0
    queued = 0
    traversed = 0
    
    # pop a state with minimal f-value from O
    while not len(open_set_heap) == 0:
        curr = heapq.heappop(open_set_heap)
        if not curr.trust:
            # print("recalculate heuristic")
            h, x, duration = __compute_heuristic_regular_cost(sync_net, curr.m, final_m, cost_function)
            
            number_solved_lps += 1
            duration_solving_lps += duration
            
            tp = SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True)
            heapq.heappush(open_set_heap, tp)
            heapq.heapify(open_set_heap)  # transform a populated list into a heap
            continue

        visited += 1
        current_marking = curr.m
        print(current_marking)
        print(final_m)

        # !!! For offline CC -> we finish the search if both the trace and the model part final markings are reached (?)
        if complete_trace:
            print(complete_trace)
            trace_final = False
            model_final = False
            for place in current_marking:
                for place2 in final_m:
                    if p_utils.place_from_synchronous_product_net_belongs_to_trace_net_part(place) and \
                       p_utils.place_from_synchronous_product_net_belongs_to_trace_net_part(place2) and \
                       place.name == place2.name:
                        trace_final = True
                    if p_utils.place_from_synchronous_product_net_belongs_to_process_net_part(place) and \
                       p_utils.place_from_synchronous_product_net_belongs_to_process_net_part(place2) and \
                       place.name == place2.name:
                        model_final = True
            print(trace_final)
            print(model_final)
            if trace_final and model_final:
                print("final")
                return __get_alignment_info(curr, visited, queued, traversed), \
                    open_set_heap, closed_set, duration_solving_lps, number_solved_lps                
        else:
        # check if we reached the final marking
            for place in current_marking:
                if p_utils.place_from_synchronous_product_net_belongs_to_trace_net_part(place):
                    for place2 in final_m:
                        if p_utils.place_from_synchronous_product_net_belongs_to_trace_net_part(place2):
                            if place.name == place2.name:
                                
                                # found a final marking of the trace net --> put marking back in open set
                                heapq.heappush(open_set_heap, curr)
                                
                                return __get_alignment_info(curr, visited, queued, traversed), \
                                       open_set_heap, closed_set, duration_solving_lps, number_solved_lps

        closed_set.add(current_marking)

        # Check if synchronous move is possible
        sync_move_possible = False
        sync_t = None
        for t in petri.semantics.enabled_transitions(sync_net, current_marking):
            if t.label[0] == t.label[1] and incremental_trace[-1][activity_key] == t.label[0]:
                sync_move_possible = True
                sync_t = t
        
        # Check if correct synchronous move is possible
        corr_sync_move_possible = False
        if sync_move_possible:
            print("sync_move_possible")
            new_marking = petri.semantics.execute(sync_t, sync_net, current_marking)
            temp_st = SearchTuple(0, (0,0), 0, new_marking, curr, sync_t, x, True)
            temp_st.current_variable_values = curr.current_variable_values.copy()
            corr_sync_move_possible = True

            # get the prime variables (if there are)
            prime_variables = {}
            tr = p_utils.get_transition_by_name(petri_net, sync_t.name[1])
            if tr.write_vars != None:
                for var_name in tr.write_vars:
                    # expecting that variable name = attribute name
                    prime_variables[var_name] = incremental_trace[-1][var_name]
                    temp_st.current_variable_values[var_name] = prime_variables[var_name]
            
            if tr.guard != None:
                guard = tr.guard
                guard = guard.replace("&&","and").replace("||","or")
                for pv, pv_value in prime_variables.items():
                    if mu.represents_number(str(pv_value)):
                        guard = guard.replace(pv+"'", str(pv_value))
                    else:
                        guard = guard.replace(pv+"'", "\""+pv_value+"\"")
                for v, v_value in curr.current_variable_values.items():
                    if mu.represents_number(str(v_value)):
                        guard = guard.replace(v, str(v_value))
                    else:
                        guard = guard.replace(v, "\""+v_value+"\"")                            
                                  
                try:
                    corr_sync_move_possible = eval(guard)
                except (SyntaxError, NameError) as error:
                    corr_sync_move_possible = False
            
            # If correct synchronous move is possible 
            if corr_sync_move_possible:
                print("corr_sync_move_possible")
                # append the alignment with the new move
                prefix_alignment = curr.prefix_alignment
                prefix_alignment.append({"marking_before_transition": temp_st.p.m,
                  "label": temp_st.t.label,
                  "name": temp_st.t.name,
                  "marking_after_transition": temp_st.m,
                  "attribute_values": __get_attribute_values(incremental_trace[-1], case_key, activity_key),
                  "variable_assignments": prime_variables})
            
                g_cf = curr.g[0] + cost_function[sync_t]
                g_var = curr.g[1]
                g = (g_cf, g_var)
                g_sum = g[0] + g[1]
                
                # enum is a tuple (int, SearchTuple), alt is a SearchTuple
                alt = next((enum[1] for enum in enumerate(open_set_heap) if enum[1].m == new_marking), None)
                if alt is not None:              
                    if g_sum >= (alt.g[0]+alt.g[1]):
                        continue
                    open_set_heap.remove(alt)
                    heapq.heapify(open_set_heap)
                queued += 1
    
                duration = 0
                if dijkstra:
                    h, x, duration = 0, None, 0
                else:
                    if derive_heuristic:
                        h, x = __derive_heuristic(cost_function, sync_t, curr.h, curr.x)
                    if not h or not derive_heuristic:
                        h, x, duration = __compute_heuristic_regular_cost(sync_net, new_marking, final_m, cost_function)
                        number_solved_lps += 1
                duration_solving_lps += duration
    
                tp = SearchTuple(g_sum + h, g, h, new_marking, curr, sync_t, x, True)
                tp.prefix_alignment = prefix_alignment
                tp.current_variable_values = temp_st.current_variable_values
                heapq.heappush(open_set_heap, tp)
                heapq.heapify(open_set_heap)

        # If correct synchronous move is NOT possible -> investigate successor states
        if not corr_sync_move_possible:   
            print("corr_sync_move_not_possible")
            for t in petri.semantics.enabled_transitions(sync_net, current_marking):              
                if curr.t is not None and utils.__is_log_move(curr.t, skip) and utils.__is_model_move(t, skip):   
                    continue
    
                traversed += 1
                new_marking = petri.semantics.execute(t, sync_net, current_marking)
                if new_marking in closed_set:
                    continue
                
                temp_st = SearchTuple(0, (0,0), 0, new_marking, curr, sync_t, x, True)
                temp_st.current_variable_values = curr.current_variable_values.copy()
                temp_st.t = t
                prefix_alignment, g_var, current_variable_values = __get_mppa(petri_net, incremental_trace, temp_st, case_key, activity_key, alignment_cost, variable_cost)
                
                if prefix_alignment != None:            
    
                    g_cf = curr.g[0] + cost_function[t]
                    g = (g_cf, g_var)
                    g_sum = g[0] + g[1]
        
                    # enum is a tuple (int, SearchTuple), alt is a SearchTuple
                    alt = next((enum[1] for enum in enumerate(open_set_heap) if enum[1].m == new_marking), None)
                    if alt is not None:              
                        if g_sum >= (alt.g[0]+alt.g[1]):
                            continue
                        open_set_heap.remove(alt)
                        heapq.heapify(open_set_heap)
                    queued += 1
        
                    duration = 0
                    if dijkstra:
                        h, x, duration = 0, None, 0
                    else:
                        if derive_heuristic:
                            h, x = __derive_heuristic(cost_function, t, curr.h, curr.x)
                        if not h or not derive_heuristic:
                            h, x, duration = __compute_heuristic_regular_cost(sync_net, new_marking, final_m, cost_function)
                            number_solved_lps += 1
                    duration_solving_lps += duration
        
                    tp = SearchTuple(g_sum + h, g, h, new_marking, curr, t, x, True)
                    tp.prefix_alignment = prefix_alignment
                    tp.current_variable_values = current_variable_values
                    heapq.heappush(open_set_heap, tp)
                    heapq.heapify(open_set_heap)
                    

def __get_mppa(petri_net, incremental_trace, curr, case_key, activity_key, alignment_cost, variable_cost):
    # UPDATE: cost so far -> result of the varriable assignment
    # for getting the OVA, the variable writings and the guard functions are needed
    global ova_cache

    current_variable_values = curr.current_variable_values
    prefix_alignment = __reconstruct_alignment(curr)
    
    all_written_values = []
    event_iterator = iter(incremental_trace)
    event = None
    
    # get variable writings
    for index, move in enumerate(prefix_alignment):
        if move['label'][0] != SKIP:
            event = next(event_iterator)
        if move['label'][1] != SKIP: 
            tr = p_utils.get_transition_by_name(petri_net, move['name'][1])
            if tr.write_vars != None and move['label'][0] != SKIP:
                written_values = dict()
                for var_name in tr.write_vars:
                    var = p_utils.get_variable_by_name(petri_net, var_name)
                    written_values[var.name] = event[var.name]
                all_written_values.append(written_values)
            else:
                all_written_values.append(None)
  
    tr_list = []
    for move in prefix_alignment:
        if str(move["label"][1]) != SKIP:
            tr_list.append(str(move["name"][1]))
        
    str_tr_list = str(tr_list)
    str_all_written_values = str(all_written_values)
    
    # check if the problem is in the OVA cache
    if (str_tr_list, str_all_written_values) in ova_cache.keys():
        print("OVA cache used")
        ovas = ova_cache[(str_tr_list, str_all_written_values)]["ovas"]
        g_var = ova_cache[(str_tr_list, str_all_written_values)]["g_var"]
        current_variable_values = ova_cache[(str_tr_list, str_all_written_values)]["current_variable_values"]
        attr_value_deviations = ova_cache[(str_tr_list, str_all_written_values)]["attr_value_deviations"]
        
        event_iterator = iter(incremental_trace)
        event = None       
        log_step_index = 0
        process_step_index = 0
        
        # get multi-perspective prefix-alignment
        for alg_move_index, move in enumerate(prefix_alignment):
            # Add attribute values
            if move['label'][0] != SKIP:
                event = next(event_iterator)
                prefix_alignment[alg_move_index]['attribute_values'] = __get_attribute_values(event, case_key, activity_key)
                log_step_index += 1
            # Add variable values
            if move['label'][1] != SKIP:
                tr = p_utils.get_transition_by_name(petri_net, move['name'][1])
                if tr.write_vars != None:
                    prefix_alignment[alg_move_index]['variable_assignments'] = ovas[process_step_index]
                    if move['label'][0] != SKIP and attr_value_deviations[process_step_index] != None:
                        prefix_alignment[alg_move_index]['deviations'] = attr_value_deviations[process_step_index]
                process_step_index += 1
        
        return prefix_alignment, g_var, current_variable_values
    else:
        print("MILP problem created")
        # CREATE MILP PROBLEM
        milp = mu.MILPProblemGenerator()
        process_step_milp_vars = []
        event_iterator = iter(incremental_trace)
        event = None
        
        # set initial values for the variables
        for var in petri_net.variables:
            milp.add_variable(var.name, var.dom, var.init_val)
          
        # get variable writings and guard functions
        for index, move in enumerate(prefix_alignment):
            # if sync./log move 
            if move['label'][0] != SKIP:
                event = next(event_iterator)
            # if sync./model move
            if move['label'][1] != SKIP:
                tr = p_utils.get_transition_by_name(petri_net, move['name'][1])
                if tr.write_vars != None:
                    # sync. move
                    if move['label'][0] != SKIP:
                        milp_vars = set()
                        for var_name in tr.write_vars:
                            var = p_utils.get_variable_by_name(petri_net, var_name)
                            attr_value = event[var.name]
                            try:
                                milp_var_name, var_value = milp.add_variable_writing(var.name, var.dom, var.init_val, attr_value, variable_cost[var.name][1])
                                milp_vars.add(milp_var_name)
                            except KeyError:
                                print("Error: The event has no "+var.name+" attribute!")
                        process_step_milp_vars.append(milp_vars)
                    # model move
                    elif move['label'][0] == SKIP:
                        milp_vars = set()
                        for var_name in tr.write_vars:
                            var = p_utils.get_variable_by_name(petri_net, var_name)
                            milp_var_name = milp.add_variable(var.name, var.dom, var.init_val)
                            milp_vars.add(milp_var_name)
                        process_step_milp_vars.append(milp_vars)
                else:
                    process_step_milp_vars.append(None)
                if tr.guard != None:
                    milp.add_guard_function(tr.guard)
        
        # SOLVE MILP PROBLEM
        
        # milp.print_model()
        milp.solve_problem(output_path, incremental_trace[-1][case_key], len(incremental_trace))
        # milp.print_solution()
        
        if not milp.is_solution_optimal():
            return None, None, None
        
        # RECONCSTRUCT MULTI-PERSPECTIVE PREFIX-ALIGNMENT
        
        # get OVAs and current variable assignments
        process_step_variable_assignments = []
        for milp_vars in process_step_milp_vars:
            if milp_vars != None:
                variable_assignments = {}
                for milp_var_name in milp_vars:
                    var_name = "_".join(str.split(milp_var_name,"_")[0:-1])
                    var_type = p_utils.get_variable_by_name(petri_net, var_name).dom
                    var_value = milp.get_variable_assignment(milp_var_name, var_type)
                    variable_assignments[var_name] = var_value
                    current_variable_values[var_name] = var_value
                process_step_variable_assignments.append(variable_assignments)
            else:
                process_step_variable_assignments.append(None)
        
        event_iterator = iter(incremental_trace)
        event = None       
        log_step_index = 0
        process_step_index = 0
        g_var = 0
        process_step_deviations = []
        
        # get prefix-alignment and g
        for alg_move_index, move in enumerate(prefix_alignment):
            # Add attribute values
            if move['label'][0] != SKIP:
                event = next(event_iterator)
                prefix_alignment[alg_move_index]['attribute_values'] = __get_attribute_values(event, case_key, activity_key)
                log_step_index += 1
            # Add variable values
            if move['label'][1] != SKIP:
                tr = p_utils.get_transition_by_name(petri_net, move['name'][1])
                deviations = []
                if tr.write_vars != None:
                    prefix_alignment[alg_move_index]['variable_assignments'] = process_step_variable_assignments[process_step_index]                  
                    # sync. move
                    if move['label'][0] != SKIP:
                        for var_name, var_value in process_step_variable_assignments[process_step_index].items():
                            if var_name in prefix_alignment[alg_move_index]["attribute_values"]:
                                attr_value = prefix_alignment[alg_move_index]["attribute_values"][var_name]
                                if type(attr_value) != type(var_value) and type(attr_value) == str:
                                    var_type = p_utils.get_variable_by_name(petri_net, var_name).dom
                                    attr_value = mu.convert_value(var_type, attr_value)
                                if attr_value != var_value:
                                    g_var += variable_cost[var_name][1] # incorrect value cost
                                    deviations.append((var_name, "incorrect value"))
                            else:
                                g_var += variable_cost[var_name][0] # missing value cost
                                deviations.append((var_name, "missing value"))
                        # If there are deviations (i.e., it's an incorrect sync. move)
                        if len(deviations) > 0:
                            prefix_alignment[alg_move_index]['deviations'] = deviations
                    # model move
                    elif move['label'][0] == SKIP:
                        for var_name, var_value in process_step_variable_assignments[process_step_index].items():
                            g_var += variable_cost[var_name][0]     # missing value cost
                            deviations.append((var_name, "missing value"))
                if len(deviations) > 0:
                    process_step_deviations.append(deviations)
                else:
                    process_step_deviations.append(None)
                process_step_index += 1
        
        
        # put the values in the OVA cache
        ova_cache[(str_tr_list, str_all_written_values)] = {}
        ova_cache[(str_tr_list, str_all_written_values)]["ovas"] = process_step_variable_assignments
        ova_cache[(str_tr_list, str_all_written_values)]["g_var"] = g_var
        ova_cache[(str_tr_list, str_all_written_values)]["current_variable_values"] = current_variable_values
        ova_cache[(str_tr_list, str_all_written_values)]["attr_value_deviations"] = process_step_deviations
        
        return prefix_alignment, g_var, current_variable_values

def __reconstruct_alignment(state):
    # state is a SearchTuple
    parent = state.p
    alignment = [{"marking_before_transition": state.p.m,
                  "label": state.t.label,
                  "name": state.t.name,
                  "marking_after_transition": state.m}]
    while parent.p is not None:
        alignment = [{"marking_before_transition": parent.p.m,
                      "label": parent.t.label,
                      "name": parent.t.name,
                      "marking_after_transition": parent.m}] + alignment
        parent = parent.p
    return alignment

def __get_alignment_info(state, visited, queued, traversed):
    # state is a SearchTuple
    return {'alignment': state.prefix_alignment, 'cost': state.g, 'visited_states': visited, 'queued_states': queued,
            'traversed_arcs': traversed}

def __compute_heuristic_regular_cost(sync_net, current_marking, final_marking, costs):
    start_time = time.time()
    solver = pywraplp.Solver('LP', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    variables = {}
    constraints = []
    for t in sync_net.transitions:
        if costs[t] < math.inf:
            # only create variables that have finite cost/ probability > 0
            variables[t] = solver.NumVar(0, solver.infinity(), str(t.name))
    # calculate current number of tokens in the process net part of the synchronous product net
    number_tokens_in_process_net_part = 0
    for p in current_marking:
        if p.name[0] == SKIP:
            number_tokens_in_process_net_part += current_marking[p]

    # constraint that enforces that at least one token is in the process net part of the synchronous product net
    # example: 1 <= var1 * coefficient1 + var2 * coefficient2 + ... + constant
    # rewrite to -->  1 - constant <= var1 * coefficient1 + var2 * coefficient2 + ...
    lb = 1 - number_tokens_in_process_net_part
    constraint_one_token_in_process_net_part = solver.Constraint(lb, solver.infinity())
    # store coefficients for each variable here because when calling constraint.SetCoefficient multiple times for the
    # same variable it overwrites always the last value for the given variable, i.e. it is NOT possible to model the
    # following constraint: x >= x1 + x2 -x1 with:
    # c.SetCoefficient(x1 , 1)
    # c.SetCoefficient(x2 , 1)
    # c.SetCoefficient(x1 , -1) --> overwrites the previous coefficient of x1
    constraint_one_token_in_process_net_part_coefficients = {}
    for v in variables:
        constraint_one_token_in_process_net_part_coefficients[v] = 0

    # define constraints
    for p in sync_net.places:
        arcs_to_transitions = []  # list of all transitions that have an incoming arc from the current place
        arcs_from_transitions = []  # list of all transitions that have an arc pointing to the current place

        for out_arc in p.out_arcs:
            arcs_to_transitions.append(out_arc.target)

        for in_arc in p.in_arcs:
            arcs_from_transitions.append(in_arc.source)

        if p.name[1] == SKIP:
            # place belongs to the trace net part
            lb_and_ub = final_marking[p] - current_marking[p]
            c = solver.Constraint(lb_and_ub, lb_and_ub)
        else:
            # place belongs to the process net part
            # enforce that the constraint is greater or equal 0, i.e.,
            # constraint + constant >= 0  -->  constraint >= 0 - constant
            c = solver.Constraint(0 - current_marking[p], solver.infinity())

            for t in arcs_to_transitions:
                if t in variables:
                    constraint_one_token_in_process_net_part_coefficients[t] -= 1

            for t in arcs_from_transitions:
                if t in variables:
                    constraint_one_token_in_process_net_part_coefficients[t] += 1

        for t in arcs_to_transitions:
            if t in variables:
                c.SetCoefficient(variables[t], -1)
        for t in arcs_from_transitions:
            if t in variables:
                c.SetCoefficient(variables[t], 1)
        constraints.append(c)
    # build constraint that enforces at least one token in the process net part
    for v in variables:
        constraint_one_token_in_process_net_part.SetCoefficient(variables[v],
                                                                constraint_one_token_in_process_net_part_coefficients[v]
                                                                )
    objective = solver.Objective()
    for v in variables:
        objective.SetCoefficient(variables[v], costs[v])
    objective.SetMinimization()
    solver.Solve()
    # debugging
    # print('Number of variables =', solver.NumVariables())
    # print('Number of constraints =', solver.NumConstraints())
    # print('Solution:')
    # for v in variables:
    #     print(str(v.name) + ":" + str(variables[v].solution_value()))
    lp_solution = 0
    res_vector = {}
    for v in variables:
        lp_solution += variables[v].solution_value() * costs[v]
        res_vector[v] = variables[v].solution_value()
    duration = time.time() - start_time
    return lp_solution, res_vector, duration


def __derive_heuristic(costs, transition, heuristic_value, res_vector):
    if res_vector[transition] > 0:
        new_res_vector = res_vector.copy()
        new_res_vector[transition] -= 1
        new_heuristic_value = heuristic_value - costs[transition]
        return new_heuristic_value, new_res_vector
    return None, None


class SearchTuple:
    def __init__(self, f, g, h, m, p, t, x, trust):
        self.f = f
        # g is a tuple -> (g_cf, g_var)
        self.g = g
        self.h = h
        self.m = m
        self.p = p
        self.t = t
        self.x = x
        self.trust = trust
        # extensions
        self.current_variable_values = dict()
        self.prefix_alignment = []

    def __lt__(self, other):
        if self.f < other.f:
            return True
        elif other.f < self.f:
            return False
        else:
            return self.h < other.h

    def __get_firing_sequence(self):
        ret = []
        if self.p is not None:
            ret = ret + self.p.__get_firing_sequence()
        if self.t is not None:
            ret.append(self.t)
        return ret

    def __repr__(self):
        string_build = ["\nm=" + str(self.m), " f=" + str(self.f), ' g=' + str(self.g), " h=" + str(self.h),
                        " path=" + str(self.__get_firing_sequence()) + "\n\n"]
        return " ".join(string_build)
