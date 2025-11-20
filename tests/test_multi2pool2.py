#!/usr/bin/env python3
"""
Script de prueba para multi2pool2.py
Procesa solo S01-DMT para verificar que el pipeline funciona
"""

import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

# ============================================================
# Funciones auxiliares
# ============================================================

def save_file(data, folder, file):
    with open(folder+file+".pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_file(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

def network_filter(phase, df_labels, hemi="both", net="all"):
    df = pd.concat([pd.DataFrame(phase), df_labels], axis=1)
    df = (df[df["hemi"]==hemi] if hemi != "both" else df)
    df = (df[df["net"]==net] if net != "all" else df)
    df = df.drop(columns=["hemi","label","net"])
    return df

def order_parameter(phase):
    r = np.abs(np.exp(1j*phase).mean(axis=0))
    return r

# ============================================================
# Configuración
# ============================================================

path = "/home/tomas/PycharmProjects/dmt_fz/dmt/fwd-inv-stc/"
band_list = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
hemi_list = ["RH", "LH", "both"]
net_list = ["FPN", "DMN", "DAN", "LN ", "SVA", "SMN", "VN "]

# Cargar metadata
print("Cargando metadata (extra.pkl)...")
[node_colors, label_names, label_names_short, stc_coords_3d, 
 ch_names, mapping, eeg_coords_2d] = load_file(path + "extra.pkl")

# Crear DataFrame con info de las parcelas
df_labels = pd.DataFrame()
df_labels["label"] = [x[:6] for x in label_names]
df_labels["hemi"] = [x[:2] for x in df_labels["label"]]
df_labels["net"] = [x[3:] for x in df_labels["label"]]

print(f"✓ {len(label_names)} parcelas cargadas")
print(f"  Redes: {df_labels['net'].unique().tolist()}")
print(f"  Hemisferios: {df_labels['hemi'].unique().tolist()}")

# ============================================================
# Procesamiento
# ============================================================

def do_the_math2(file_name):
    """
    Procesa un archivo de fases y genera order_all por redes/hemisferios
    """
    data_dict = load_file(file_name)
    
    # Extraer nombre y condición
    import os
    fname = os.path.basename(file_name)
    subject_id = fname.replace("phases", "order_all").replace(".pkl", "")
    cond = os.path.basename(os.path.dirname(file_name))
    output_folder = os.path.dirname(file_name) + "/"
    
    print(f"\n{'='*60}")
    print(f"Procesando: {fname} (condición: {cond})")
    print(f"{'='*60}")
    
    r_dict = {}
    r_dict[cond] = {}
    
    for band in band_list:
        epoch_list = data_dict["phases_stc"][band]
        r_dict[cond][band] = {}
        print(f"\nBanda {band} - {len(epoch_list)} épocas")
        
        for hemi in hemi_list:
            r_dict[cond][band][hemi] = {}
            
            for net in net_list:
                r_dict[cond][band][hemi][net] = []
            
                for epoch_data in tqdm(epoch_list, desc=f"  {hemi} {net.strip()}", leave=False):
                    filtered = network_filter(epoch_data, df_labels, hemi=hemi, net=net)
                    # NO calcula order_parameter aquí, solo filtra
                    r_dict[cond][band][hemi][net].append(filtered)
    
    output_path = output_folder + subject_id + ".pkl"
    save_file(r_dict, output_folder, subject_id)
    print(f"\n✓ Guardado: {output_path}")
    return r_dict

# ============================================================
# Ejecutar con archivo de prueba
# ============================================================

if __name__ == '__main__':
    test_file = path + "DMT/phases-S01-DMT.pkl"
    
    print("="*60)
    print("TEST: Procesando S01-DMT")
    print("="*60)
    
    result = do_the_math2(test_file)
    
    print("\n" + "="*60)
    print("✓ Procesamiento completado!")
    print("="*60)
    print(f"\nArchivo generado:")
    print(f"  {path}DMT/order_all-S01-DMT.pkl")
    print(f"\nEstructura de datos:")
    print(f"  - Condición: {list(result.keys())}")
    print(f"  - Bandas: {list(result['DMT'].keys())}")
    print(f"  - Hemisferios: {list(result['DMT']['Delta'].keys())}")
    print(f"  - Redes: {list(result['DMT']['Delta']['both'].keys())}")






