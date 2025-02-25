# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:31:26 2025

@author: masol
"""

"""
RED DE TAXONOMÍA DE PRODUCTOS DEL PARAGUAY (2013-2022)

"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Cargar la base de datos global (exportaciones mundiales 2022)
file_path = r"promedio_exportaciones_filtrado.xlsx"
EXP_df = pd.read_excel(file_path, header=None)

# Extraer country_id y product_id
country_ids = EXP_df.iloc[1:, 0].values  # Primera columna (ignorar encabezado)
product_ids = EXP_df.iloc[0, 1:].values  # Primera fila (ignorar encabezado)
EXP = EXP_df.iloc[1:, 1:].values.astype(float)  # Matriz de exportaciones ignorando encabezados

# Cálculo de VCR y Mcp
Vc = np.sum(EXP, axis=1)  # Diversidad de cada país (dc)
Vp = np.sum(EXP, axis=0)  # Ubicuidad de cada producto (up)
SumTotal = np.sum(Vc)
VCR = (EXP / Vc[:, None]) / (Vp / SumTotal)
Mcp = (VCR >= 1).astype(int)

# Cálculo de la matriz Bpp'
ubiquity = np.sum(Mcp, axis=0)  # up
diversity = np.sum(Mcp, axis=1)  # dc

Bpp_prime = np.zeros((Mcp.shape[1], Mcp.shape[1]))  # Inicializar matriz

for p in range(Mcp.shape[1]):
    for p_prime in range(Mcp.shape[1]):
        if p != p_prime:  # Evitar calcular auto-conexiones
            numerator = np.sum(Mcp[:, p] * Mcp[:, p_prime] / diversity)
            denominator = max(ubiquity[p], ubiquity[p_prime])
            if denominator != 0:  # Evitar divisiones por cero
                Bpp_prime[p, p_prime] = numerator / denominator

# Reescalar Bpp' entre 0 y 1
Bpp_prime_rescaled = (Bpp_prime - Bpp_prime.min()) / (Bpp_prime.max() - Bpp_prime.min())
Bpp_prime_rescaled = np.round(Bpp_prime_rescaled, decimals=10)

# Crear red de taxonomía únicamente para Paraguay (country_id = 600)
def plot_taxonomy_hierarchy_paraguay(Bpp, product_ids, final_complexity, Mcp, country_id, threshold):
    G = nx.DiGraph()

    # Seleccionar la fila de Mcp correspondiente a Paraguay
    paraguay_idx = np.where(country_ids == country_id)[0][0]
    paraguay_products = Mcp[paraguay_idx, :]  # Productos asociados a Paraguay

    # Agregar nodos basados en productos exportados por Paraguay
    for i in range(len(product_ids)):
        if final_complexity[i] > 0:
            color = "green" if paraguay_products[i] == 1 else "red"
            G.add_node(product_ids[i], size=final_complexity[i], color=color)

    # Agregar aristas basadas en Bpp (del producto menos complejo al más complejo)
    for i in range(len(product_ids)):
        for j in range(len(product_ids)):
            if (
                Bpp[i, j] > threshold
                and final_complexity[i] < final_complexity[j]
                and paraguay_products[i] > 0
            ):
                G.add_edge(product_ids[i], product_ids[j], weight=Bpp[i, j])

    # Eliminar conexiones redundantes
    for node in list(G.nodes):
        predecessors = list(G.predecessors(node))
        for p1 in predecessors:
            for p2 in predecessors:
                if p1 != p2 and G.has_edge(p1, node) and G.has_edge(p2, node):
                    if G.has_edge(p1, p2):
                        G.remove_edge(p1, node)

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print("El grafo no tiene nodos o aristas suficientes para visualizar.")
        return G

    print(f"Grafo generado con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")

    # Visualizar el grafo
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
    plt.figure(figsize=(12, 12))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=[G.nodes[n]['size'] * 1000 for n in G.nodes],
        font_size=8,
        edge_color="gray",
        arrowsize=20,
        node_color=[G.nodes[n]['color'] for n in G.nodes],
        width=[G[u][v]['weight'] * 10 for u, v in G.edges]
    )
    plt.title(f"Red Dirigida de Taxonomía de Productos - Paraguay")
    plt.show()
    
    return G

# Cálculo de Fitness y Complejidad
N = 1000  # Número de iteraciones
FcN = np.ones((Mcp.shape[0], N))
QpN = np.ones((Mcp.shape[1], N))
for k in range(1, N):
    FcN[:, k] = np.sum(Mcp * QpN[:, k - 1], axis=1)
    FcN[:, k] /= np.mean(FcN[:, k])
    QpN[:, k] = 1 / np.sum(Mcp * (1 / FcN[:, k - 1])[:, None], axis=0)
    QpN[:, k] /= np.mean(QpN[:, k])

final_fitness = FcN[:, -1]
final_complexity = QpN[:, -1]

# Definir umbral dinámico para conexiones
threshold = np.percentile(Bpp_prime_rescaled[Bpp_prime_rescaled > 0], 90)

# Generar la red y obtener el grafo
G = plot_taxonomy_hierarchy_paraguay(Bpp_prime_rescaled, product_ids, final_complexity, Mcp, country_id=600, threshold=threshold)

# Guardar resultados en un archivo Excel
output_file = "Resultados_Analisis_Economico_Paraguay_2013a2022.xlsx"
with pd.ExcelWriter(output_file) as writer:
    pd.DataFrame(VCR, index=country_ids, columns=product_ids).to_excel(writer, sheet_name="VCR")
    pd.DataFrame(Mcp, index=country_ids, columns=product_ids).to_excel(writer, sheet_name="Mcp")
    pd.DataFrame(Bpp_prime_rescaled, index=product_ids, columns=product_ids).to_excel(writer, sheet_name="Bpp")
    pd.DataFrame({"country_id": country_ids, "fitness": final_fitness}).to_excel(writer, sheet_name="Fitness", index=False)
    pd.DataFrame({"product_id": product_ids, "complexity": final_complexity}).to_excel(writer, sheet_name="Complexity", index=False)

# Exportar nodos y aristas para Gephi
nodes_file = "nodos_original.csv"
edges_file = "aristas_original.csv"

nodes_data = [{"id": n, "size": G.nodes[n]["size"], "color": G.nodes[n]["color"]} for n in G.nodes]
edges_data = [{"source": u, "target": v, "weight": G[u][v]["weight"]} for u, v in G.edges]

pd.DataFrame(nodes_data).to_csv(nodes_file, index=False)
pd.DataFrame(edges_data).to_csv(edges_file, index=False)

print(f"Nodos exportados a {nodes_file}")
print(f"Aristas exportadas a {edges_file}")
