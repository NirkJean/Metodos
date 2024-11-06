# Asegúrate de instalar las librerías necesarias: streamlit, pulp, matplotlib
# !pip install streamlit pulp matplotlib

import streamlit as st
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpInteger, value
import matplotlib.pyplot as plt
import networkx as nx

# Función principal para la aplicación en Streamlit
def main():
    st.title("Solucionador de Branch and Bound - Método de Dakin")
    st.write("Resuelve el problema paso a paso usando el Método Branch and Bound de Dakin.")
    
    # Definición del problema
    obj_coeffs = [4, 3, 3]  # Coeficientes para la función objetivo
    restricciones = [
        {'coeffs': [4, 2, 1], 'bound': 10},
        {'coeffs': [3, 4, 2], 'bound': 14},
        {'coeffs': [2, 1, 3], 'bound': 7}
    ]
    
    # Mostrar el enunciado del problema
    st.subheader("Enunciado del Problema")
    st.write("Maximizar: P(x1, x2, x3) = 4x1 + 3x2 + 3x3")
    st.write("Sujeto a:")
    st.write("4x1 + 2x2 + x3 ≤ 10")
    st.write("3x1 + 4x2 + 2x3 ≤ 14")
    st.write("2x1 + x2 + 3x3 ≤ 7")
    st.write("donde x1, x2, x3 son enteros no negativos.")
    
    # Definir variables
    def branch_and_bound(obj_coeffs, restricciones):
        decision_tree = nx.DiGraph()
        node_counter = 1
        solucion_optima = None
        valor_optimo = -float("inf")
        subproblemas = [(LpProblem(f"Problema_{node_counter}", LpMaximize), None)]

        while subproblemas:
            prob, parent_node = subproblemas.pop(0)
            x1 = LpVariable("x1", lowBound=0, cat=LpInteger)
            x2 = LpVariable("x2", lowBound=0, cat=LpInteger)
            x3 = LpVariable("x3", lowBound=0, cat=LpInteger)
            
            # Definir la función objetivo
            prob += lpSum([c * x for c, x in zip(obj_coeffs, [x1, x2, x3])])

            # Agregar restricciones
            for restriccion in restricciones:
                prob += lpSum([c * x for c, x in zip(restriccion['coeffs'], [x1, x2, x3])]) <= restriccion['bound']
            
            prob.solve()
            solucion = [value(x1), value(x2), value(x3)]
            valor_solucion = value(prob.objective)

            if valor_solucion > valor_optimo:
                solucion_optima = solucion
                valor_optimo = valor_solucion

            # Ramificar en la primera variable fraccional (si existe)
            var_fraccional = next((var for var in [x1, x2, x3] if int(value(var)) != value(var)), None)
            
            if var_fraccional is None:
                decision_tree.add_node(node_counter, solution=solucion, objective=valor_solucion, leaf=True)
                node_counter += 1
            else:
                decision_tree.add_node(node_counter, solution=solucion, objective=valor_solucion, leaf=False)
                lower_bound = int(value(var_fraccional))
                prob_lower = prob.copy()
                prob_lower += var_fraccional <= lower_bound
                
                prob_upper = prob.copy()
                prob_upper += var_fraccional >= lower_bound + 1

                subproblemas.append((prob_lower, node_counter))
                subproblemas.append((prob_upper, node_counter))
                node_counter += 1
        
        return solucion_optima, valor_optimo, decision_tree

    solucion_optima, valor_optimo, decision_tree = branch_and_bound(obj_coeffs, restricciones)
    st.write("**Solución Óptima**")
    st.write(f"x1 = {solucion_optima[0]}, x2 = {solucion_optima[1]}, x3 = {solucion_optima[2]}")
    st.write(f"Valor Óptimo de la Función Objetivo: {valor_optimo}")

    # Visualización del Árbol de Decisiones
    st.subheader("Árbol de Decisiones")
    fig, ax = plt.subplots()
    pos = nx.spring_layout(decision_tree)
    nx.draw(decision_tree, pos, with_labels=True, ax=ax, node_size=700, node_color="lightblue", font_size=10)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
