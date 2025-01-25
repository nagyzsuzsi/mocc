'''
    This file is NOT part of PM4Py.
    Author: Zsuzsanna Nagy
'''

from __future__ import print_function
from pulp import *
import re
import pandas as pd


def represents_number(s):
    try: 
        if s == "True" or s == "False":
            return True
        float(s)
        return True
    except ValueError:
        return False

def convert_value(val_type, str_value):
    if val_type == int:
        return int(str_value)
    if val_type == float:
        return float(str_value)
    if val_type == bool:
        return bool(str_value)
    

B = {int: 100000, float: 100000.0}
e = {int: 1, float: 0.0000001}

class MILPProblemGenerator:
    
    def __init__(self):
        self.__solver = LpProblem("OVA",LpMinimize)
        self.__objective = dict()
        
        # DataFrame to store the constraint coeffs for the MILP variables (1 row = 1 atomic guard function)
        # - col 0: b
        # - col 1-n: milp variables 
        self.__df_ge_cons = pd.DataFrame()
        self.__df_ge_cons.insert(0, "b", 0)
        
        # store the MILP variables in the same order they appear in the DataFrame
        self.__guard_expr_milp_vars = dict()
        
        self.__str_values_list = []
        self.__write_vars_dict = dict()
        
        self.__milp_vars = dict()
        
        self.__or_i = 1
        self.__and_i = 1
        self.__xor_i = 1
 
    def get_df(self):
        print(self.__df_ge_cons)
 
    def __remove_outter_parentheses(self, s):
        if len(s) >=3 and s[0] == "(" and s[-1] == ")":
            return s[1:-1]
        else:
            return s
    
    # input can be str, int, float, bool
    def __convert_to_number(self, s):
        if type(s) == float:
            return s
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return None
    
    def __get_guardf_components(self, s, split_str):
        s = s[1:len(s)-1]
        return s.split(split_str)
    
    def __get_or_rels(self, rels):
        or_rel_list = []
        for x in rels:
            if "OR" in x:
                or_rel_list.append(x)
        return or_rel_list
    
    def __get_atomic_gfs_for_or_rels(self, rels, node, container):
        if "x" in node:
            container.add(rels[node])
        else:
            self.__get_atomic_gfs_for_or_rels(rels, rels[node][0], container)
            self.__get_atomic_gfs_for_or_rels(rels, rels[node][1], container)
        
    def __complex_gf_processing(self, s):       
        rels = {}
        or_rels_atomic_guards = {}
        atomic_guards_or_rels = {}      
        
        # find the atomic guard functions
        p = re.compile('\([a-zA-Z0-9-+*/=!><\'\"\_]*\)')
        search_results = re.finditer(p, s)
        i = 1
        for item in search_results:         
            if item.group(0) != "":
                atomic_gf = item.group(0).replace(" ","")
                atomic_gf = atomic_gf[1:len(atomic_gf)-1]   # remove "(" and ")"
                x_var = "x"+str(i)
                atomic_guards_or_rels[atomic_gf] = set()
                rels[x_var] = atomic_gf
                s = s.replace(item.group(0),x_var)
                i+=1
        
        # find the relationships between the atomic guard functions
        p = re.compile('\([a-zA-Z0-9&|]*\)')
        while s.find("&&") != -1 or s.find("||")!= -1:
            search_results = re.finditer(p, s)
            for item in search_results:
                if item.group(0) != "":
                    # AND relationship
                    if (item.group(0).find("&&") != -1):
                        and_rep = "AND"+str(self.__and_i)
                        s = s.replace(item.group(0), and_rep)
                        a,b = self.__get_guardf_components(item.group(0),"&&")
                        rels[and_rep] = [a,b]
                        self.__and_i+=1
                    # OR relationship
                    if (item.group(0).find("||") != -1):
                        or_rep = "OR"+str(self.__or_i)
                        s = s.replace(item.group(0), or_rep)
                        a,b = self.__get_guardf_components(item.group(0),"||")
                        rels[or_rep] = [a,b]
                        self.__or_i+=1
        
        # OR relationships -> atomic guard functions
        or_rel_list = self.__get_or_rels(rels)
        for or_rel in or_rel_list:
            or_rels_atomic_guards[or_rel] = (set(),set())
            self.__get_atomic_gfs_for_or_rels(rels, rels[or_rel][0], or_rels_atomic_guards[or_rel][0])
            self.__get_atomic_gfs_for_or_rels(rels, rels[or_rel][1], or_rels_atomic_guards[or_rel][1])
       
        # atomic guard functions -> MILP variables for OR relationships
        for or_rel in or_rels_atomic_guards:
            for g in or_rels_atomic_guards[or_rel][0]:
                atomic_guards_or_rels[g].add(or_rel+"_1")
            for g in or_rels_atomic_guards[or_rel][1]:
                atomic_guards_or_rels[g].add(or_rel+"_2")
        
        return or_rel_list, atomic_guards_or_rels
    
    def __add_milp_varibale_to_df(self, milp_var_name, milp_var):
        self.__df_ge_cons.insert(self.__df_ge_cons.shape[1], milp_var_name, 0)
        self.__guard_expr_milp_vars[self.__df_ge_cons.shape[1]-1] = milp_var

    def __add_vars_and_constraints_for_or_rels(self, or_rel_list):
        for or_rel in or_rel_list:
            # add variables for OR relationship
            or_i_1_str = or_rel+"_1"
            or_i_2_str = or_rel+"_2"
            self.__milp_vars[or_i_1_str] = LpVariable(or_i_1_str, lowBound = 0, upBound = 1, cat = 'Integer')
            self.__milp_vars[or_i_2_str] = LpVariable(or_i_2_str, lowBound = 0, upBound = 1, cat = 'Integer')
            
            self.__add_milp_varibale_to_df(or_i_1_str, self.__milp_vars[or_i_1_str])
            self.__add_milp_varibale_to_df(or_i_2_str, self.__milp_vars[or_i_2_str])
         
            # add constraint
            # or_i_1 + or_i_2 <= 1
            self.__solver += self.__milp_vars[or_i_1_str] + self.__milp_vars[or_i_2_str] <= 1

    def __add_vars_and_constraint_for_xor_rel(self):
        # add variables for XOR relationship
        xor_1_str = "xor_"+str(self.__xor_i)
        xor_2_str = "xor_"+str(self.__xor_i+1)
        self.__xor_i = self.__xor_i + 2
        self.__milp_vars[xor_1_str] = LpVariable(xor_1_str, lowBound = 0, upBound = 1, cat = 'Integer')
        self.__milp_vars[xor_2_str] = LpVariable(xor_2_str, lowBound = 0, upBound = 1, cat = 'Integer')
        
        self.__add_milp_varibale_to_df(xor_1_str, self.__milp_vars[xor_1_str])
        self.__add_milp_varibale_to_df(xor_2_str, self.__milp_vars[xor_2_str])
     
        # add constraint
        # xor_i + xor_i+1 <= 1
        self.__solver += self.__milp_vars[xor_1_str] + self.__milp_vars[xor_2_str] == 1
        
        return xor_1_str, xor_2_str

    def __get_sign(self, atomic_guard_expr_side, element):
        start_index = atomic_guard_expr_side.find(str(element))
        if start_index != 0:
            if atomic_guard_expr_side[start_index-1] == "-":
                return -1
            else:
                return 1
        else:
            return 1
    
    def __handle_str_value(self, value):
        if not represents_number(value):
            if value not in self.__str_values_list:
                self.__str_values_list += [value]
            return self.__str_values_list.index(value)
        else:
            return value
    
    def __process_atomic_guard_function(self, guard_expression_side, is_right, df_new_row_index):
        take_left = -1 if is_right else 1
        take_right = 1 if is_right else -1
        sum_list = re.split('\+|\-',guard_expression_side)
        for sum_element in sum_list:
            if sum_element != "":
                elements = sum_element.split('*')
                # one element -> only variable OR constant value
                if len(elements) == 1:
                    element = elements[0]
                    # if variable
                    if element in self.__df_ge_cons.columns:
                        sign = self.__get_sign(guard_expression_side,element) * take_left
                        self.__df_ge_cons.at[df_new_row_index, element] += sign
                    # if constant
                    else:
                        sign = self.__get_sign(guard_expression_side,element) * take_right
                        element = self.__handle_str_value(element)
                        self.__df_ge_cons.at[df_new_row_index, "b"] += sign * self.__convert_to_number(element)
                # two elements -> variable AND constant value
                elif len(elements) == 2:
                    var_i = 0
                    cons_i = 0
                    # find the variable
                    if elements[0] in self.__df_ge_cons.columns:
                        cons_i = 1
                    if elements[1] in self.__df_ge_cons.columns:
                        var_i = 1
                    sign_var = self.__get_sign(guard_expression_side,elements[var_i])
                    sign_cons = self.__get_sign(guard_expression_side,elements[cons_i])                  
                    
                    sign_total = sign_var * sign_cons * take_left
                    elements[cons_i] = self.__handle_str_value(elements[cons_i])
                    self.__df_ge_cons.at[df_new_row_index, elements[var_i]] += sign_total * self.__convert_to_number(elements[cons_i])
                # more than two elements
                elif len(elements) > 2:
                    print("Error: The guard function isn't linear!")
    
    def __add_variable_writing_to_write_vars_dict(self, var_name, var_type, var_init, written_value=None):
        if var_name not in self.__write_vars_dict.keys():
            self.__write_vars_dict[var_name] = []
        write_var_nr = len(self.__write_vars_dict[var_name])     
        
        if var_type == str:
            # give int code to the str value
            if written_value not in self.__str_values_list:
                self.__str_values_list += [written_value]
            self.__write_vars_dict[var_name] += [self.__str_values_list.index(written_value)]
        else:
            self.__write_vars_dict[var_name] += [written_value]
        return var_name + "_" + str(write_var_nr)
    
    def add_variable_writing(self, var_name, var_type, var_init, written_value, cost):
        if var_type == int or var_type == float:
            written_value = self.__convert_to_number(written_value)
        
        # check the written value
        if written_value != None and not (isinstance(written_value, float) and math.isnan(written_value)):
            
            # create the MILP constraints for the variable writing
            v_i = self.__add_variable_writing_to_write_vars_dict(var_name, var_type, var_init, written_value)
            _v_i = "_" + v_i
            b_i = written_value if var_type != str else self.__str_values_list.index(written_value)                   
            
            # Add variables
            if var_type == float:
                self.__milp_vars[v_i] = LpVariable(v_i)
                v_type = float
            elif var_type == bool:
                self.__milp_vars[v_i] = LpVariable(v_i, cat = 'Binary')
                v_type = int
            elif var_type == str:
                self.__milp_vars[v_i] = LpVariable(v_i, lowBound = 0, cat = 'Integer')
                v_type = int
            elif var_type == int:
                self.__milp_vars[v_i] = LpVariable(v_i, cat = 'Integer')
                v_type = int                
            self.__milp_vars[_v_i] = LpVariable(_v_i, cat='Binary')
            
            self.__add_milp_varibale_to_df(v_i, self.__milp_vars[v_i])
            
            # Add constraints
            # v_i - B * _v_i <= b_i
            # v_i + B * _v_i >= b_i
            self.__solver += self.__milp_vars[v_i] - B[v_type] * self.__milp_vars[_v_i] <= b_i
            self.__solver += self.__milp_vars[v_i] + B[v_type] * self.__milp_vars[_v_i] >= b_i
            
            # Add the boolean variable to the objective function
            self.__objective[self.__milp_vars[_v_i]] = cost
            return v_i, written_value
        else:
            print("Error: wrong variable value ", var_name, var_init, written_value)
            return self.add_variable_writing(var_name, var_type, var_init, var_init, cost)
    
    # used when there is only model move (i.e., missing variable writing)
    def add_variable(self, var_name, var_type, var_init):
        v_i = self.__add_variable_writing_to_write_vars_dict(var_name, var_type, var_init, var_init)
        if var_type == float:
            self.__milp_vars[v_i] = LpVariable(v_i)
        elif var_type == bool:
            self.__milp_vars[v_i] = LpVariable(v_i, cat = 'Binary')
        elif var_type == str:
            self.__milp_vars[v_i] = LpVariable(v_i, lowBound = 0, cat = 'Integer')
        elif var_type == int:
            self.__milp_vars[v_i] = LpVariable(v_i, cat = 'Integer')
        
        self.__add_milp_varibale_to_df(v_i, self.__milp_vars[v_i])        
        
        return v_i
    
    def add_guard_function(self, guard_expr):
        guard_expr = guard_expr.replace("\"","")
        # replace: 'var -> v_i, var -> v_i-1
        for var in self.__write_vars_dict.keys():
            if var in guard_expr:
                # get the current value of the variable
                v_i = var + "_" + str(len(self.__write_vars_dict[var])-1)
                # if the transition writes the variable -> prime variable
                if var+"'" in guard_expr:
                    v_i_1 = var + "_" + str(len(self.__write_vars_dict[var])-2)
                    guard_expr = guard_expr.replace(var, v_i_1)
                    guard_expr = guard_expr.replace(v_i_1+"'", v_i)
                else:
                    guard_expr = guard_expr.replace(var, v_i)        
        
        # split the guard expression into atomic elements
        or_rel_list, atomic_guards_or_rels = self.__complex_gf_processing(guard_expr)
        
        # add variables and constraints for OR relationships
        self.__add_vars_and_constraints_for_or_rels(or_rel_list)
        
        # create a MILP constraint for each atomic element
        for atomic_guard_expr in atomic_guards_or_rels:
            for rs in ["<=", ">=", "==", "!=", "<", ">"]:
                if len(atomic_guard_expr.split(rs)) == 2:
                    
                    # add new line to the df
                    self.__df_ge_cons.loc[self.__df_ge_cons.shape[0]] = 0                  
                    df_new_row_index = self.__df_ge_cons.shape[0] - 1
                    
                    # process the atomic guard function (left and right side)
                    left, right = atomic_guard_expr.split(rs)
                    self.__process_atomic_guard_function(left, False, df_new_row_index)                   
                    self.__process_atomic_guard_function(right, True, df_new_row_index)
                    
                    # transform into milp constraints
                    ## <=
                    if rs == "<=":                       
                        if len(atomic_guards_or_rels[atomic_guard_expr]) > 0:
                            for or_i_j_str in atomic_guards_or_rels[atomic_guard_expr]:
                                self.__df_ge_cons.at[df_new_row_index, or_i_j_str] = -B[int]
                        
                        constraint_expr = [self.__df_ge_cons.iat[df_new_row_index,j] * self.__guard_expr_milp_vars[j] for j in range(1,self.__df_ge_cons.shape[1])]
                        self.__solver += lpSum(constraint_expr) <= self.__df_ge_cons.at[df_new_row_index,"b"]
                        
                    ## >=
                    if rs == ">=":                      
                        if len(atomic_guards_or_rels[atomic_guard_expr]) > 0:
                            for or_i_j_str in atomic_guards_or_rels[atomic_guard_expr]:
                                self.__df_ge_cons.at[df_new_row_index, or_i_j_str] = B[int]
                                
                        constraint_expr = [self.__df_ge_cons.iat[df_new_row_index,j] * self.__guard_expr_milp_vars[j] for j in range(1,self.__df_ge_cons.shape[1])]
                        self.__solver += lpSum(constraint_expr) >= self.__df_ge_cons.at[df_new_row_index,"b"]
                    
                    ## ==
                    if rs == "==":
                        self.__df_ge_cons.loc[df_new_row_index+1] = self.__df_ge_cons.loc[df_new_row_index]
                        
                        if len(atomic_guards_or_rels[atomic_guard_expr]) > 0:
                            for or_i_j_str in atomic_guards_or_rels[atomic_guard_expr]:
                                self.__df_ge_cons.at[df_new_row_index, or_i_j_str] = -B[int]                                
                                self.__df_ge_cons.at[df_new_row_index+1, or_i_j_str] = B[int]
                        
                        constraint_expr_1 = [self.__df_ge_cons.iat[df_new_row_index,j] * self.__guard_expr_milp_vars[j] for j in range(1,self.__df_ge_cons.shape[1])]
                        constraint_expr_2 = [self.__df_ge_cons.iat[df_new_row_index+1,j] * self.__guard_expr_milp_vars[j] for j in range(1,self.__df_ge_cons.shape[1])]
                        self.__solver += lpSum(constraint_expr_1) <= self.__df_ge_cons.at[df_new_row_index,"b"]
                        self.__solver += lpSum(constraint_expr_2) >= self.__df_ge_cons.at[df_new_row_index+1,"b"]
                        
                    ## !=
                    if rs == "!=":
                        self.__df_ge_cons.loc[df_new_row_index+1] = self.__df_ge_cons.loc[df_new_row_index]

                        self.__df_ge_cons.at[df_new_row_index, "b"] -= e[int]
                        self.__df_ge_cons.at[df_new_row_index+1, "b"] += e[int]
                        
                        xor_1_str, xor_2_str = self.__add_vars_and_constraint_for_xor_rel()
                        self.__df_ge_cons.at[df_new_row_index, xor_1_str] = -B[int]                             
                        self.__df_ge_cons.at[df_new_row_index+1, xor_2_str] = B[int]
                        
                        if len(atomic_guards_or_rels[atomic_guard_expr]) > 0:
                            for or_i_j_str in atomic_guards_or_rels[atomic_guard_expr]:
                                self.__df_ge_cons.at[df_new_row_index, or_i_j_str] = -B[int]                              
                                self.__df_ge_cons.at[df_new_row_index+1, or_i_j_str] = B[int]
                        
                        constraint_expr_1 = [self.__df_ge_cons.iat[df_new_row_index,j] * self.__guard_expr_milp_vars[j] for j in range(1,self.__df_ge_cons.shape[1])]
                        constraint_expr_2 = [self.__df_ge_cons.iat[df_new_row_index+1,j] * self.__guard_expr_milp_vars[j] for j in range(1,self.__df_ge_cons.shape[1])]
                        self.__solver += lpSum(constraint_expr_1) <= self.__df_ge_cons.at[df_new_row_index,"b"]
                        self.__solver += lpSum(constraint_expr_2) >= self.__df_ge_cons.at[df_new_row_index+1,"b"]
                        
                    ## <
                    if rs == "<":
                        self.__df_ge_cons.at[df_new_row_index, "b"] -= e[int]
                        
                        if len(atomic_guards_or_rels[atomic_guard_expr]) > 0:
                            for or_i_j_str in atomic_guards_or_rels[atomic_guard_expr]:
                                self.__df_ge_cons.at[df_new_row_index, or_i_j_str] = -B[int]
                        
                        constraint_expr = [self.__df_ge_cons.iat[df_new_row_index,j] * self.__guard_expr_milp_vars[j] for j in range(1,self.__df_ge_cons.shape[1])]
                        self.__solver += lpSum(constraint_expr) <= self.__df_ge_cons.at[df_new_row_index,"b"]
                        
                    ## >
                    if rs == ">":                    
                        self.__df_ge_cons.at[df_new_row_index, "b"] += e[int]
                        
                        if len(atomic_guards_or_rels[atomic_guard_expr]) > 0:
                            for or_i_j_str in atomic_guards_or_rels[atomic_guard_expr]:
                                self.__df_ge_cons.at[df_new_row_index, or_i_j_str] = B[int]
                        
                        constraint_expr = [self.__df_ge_cons.iat[df_new_row_index,j] * self.__guard_expr_milp_vars[j] for j in range(1,self.__df_ge_cons.shape[1])]
                        self.__solver += lpSum(constraint_expr) >= self.__df_ge_cons.at[df_new_row_index,"b"]
                    
                    break 

    def solve_problem(self, output_path, case_id, nr):
        self.__solver += LpAffineExpression(self.__objective)
        
        file_name = "OVA_problem_" + str(case_id) + "_" + str(nr) + ".lp"

        self.__solver.writeLP(os.path.join(output_path, file_name))
        self.__solver.solve(PULP_CBC_CMD(msg=0, logPath=os.devnull))
        self.__status = LpStatus[self.__solver.status]
    
    def get_variable_assignment(self, milp_var_name, var_type=None):
        for var in self.__solver.variables():
            if var.name == milp_var_name:
                solution = var.varValue
                if var_type == int:
                    return int(solution)
                if var_type == str and solution<len(self.__str_values_list):
                    return self.__str_values_list[int(solution)]
                return solution
        else:
            return None
    
    def get_variable_assignments(self):
        variable_assignments = dict()
        for var in self.__solver.variables():
            if var.name[0] != "_": 
                variable_assignments[var.name] = var.varValue
        return variable_assignments
    
    def get_variable_assignment_cost(self):
        if self.__status == "Optimal":
            return value(self.__solver.objective)
        else:
            return None
    
    def is_solution_optimal(self):
        if self.__status == "Optimal":
            return True
        else:
            return False
