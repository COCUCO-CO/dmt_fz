# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 20:33:29 2025

@author: xochipilli
"""

import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from scipy import signal
import pandas as pd
import os


folder = "E:\\FedeZ\\MD\\Dinamicas\\APO\\"

# List of ligands and replicates to process
lig_list = ["5HT", "KET", "LSD", "DOI", "PSI", "MES"]
replicates = [1, 2, 3]
  # Base folder

# Results container for all analyses
all_results = []

for lig in lig_list:
    print(f"Processing ligand: {lig}")

    for rep in replicates:
        print(f"  Replicate: {rep}")

        # Define file paths
        nc_path = os.path.join(folder, lig, f"{lig}_1us{rep}-strip_cent.nc")
        top_path = os.path.join(folder, lig, f"{lig}-strip.parm7")

        # Check if files exist before processing
        if not os.path.exists(nc_path) or not os.path.exists(top_path):
            print(f"  SKIPPING: Files not found for {lig} replicate {rep}")
            continue

        # Load the trajectory
        try:
            traj = md.load(nc_path, top=top_path)
            print(f"  Loaded trajectory with {traj.n_frames} frames")
        except Exception as e:
            print(f"  ERROR loading trajectory: {e}")
            continue

        # Select ligand atoms
        ligand_indices = traj.topology.select(f'resname {lig}')
        if len(ligand_indices) == 0:
            print(f"  WARNING: Couldn't find atoms with resname {lig}")
            # Try alternate names if needed
            possible_alts = {
                "5HT": ["SER", "5-HT", "SRO"],
                "KET": ["KET", "KETA"],
                "LSD": ["LSD", "L25"],
                "DOI": ["DOI", "DOX"],
                "PSI": ["PSI", "PSIL", "PS"],
                "MES": ["MES", "MSC", "MESC"]
            }

            if lig in possible_alts:
                for alt_name in possible_alts[lig]:
                    ligand_indices = traj.topology.select(f'resname {alt_name}')
                    if len(ligand_indices) > 0:
                        print(f"  Found ligand with alternate name: {alt_name}")
                        break

        if len(ligand_indices) == 0:
            print(f"  ERROR: Couldn't identify ligand atoms for {lig}")
            continue

        protein_indices = traj.topology.select('protein')

        # Align trajectory to protein to remove overall motion
        aligned_traj = traj.superpose(traj, frame=0, atom_indices=protein_indices)

        # 1. RMSD Analysis
        # aca toma el frame 0 como referencia y calcula para cada frame la root mean square distance (cuanto se desvia cada atomo respecto de su posicion en el frame 0, despues promedia
        # todos las diatancias del frame y eso te da el valor para cada frame)
        rmsd_data = md.rmsd(aligned_traj, aligned_traj, frame=0, atom_indices=ligand_indices)

        # 2. Distance Analysis
        # Find binding site residues
        binding_site_indices = []
        for res_idx in range(traj.topology.n_residues):
            res = traj.topology.residue(res_idx)
            if res.is_protein:  # Only consider protein residues
                res_atoms = [atom.index for atom in res.atoms]
                min_dist = md.compute_neighbors(traj[0], 0.5, ligand_indices, res_atoms)
                if len(min_dist[0]) > 0:
                    binding_site_indices.extend(res_atoms)

        print(f"  Found {len(binding_site_indices)} atoms in the binding site")

        # Calculate COM distances
        distances = []
        if binding_site_indices:
            # Calculate centers of mass
            ligand_com = md.compute_center_of_mass(aligned_traj.atom_slice(ligand_indices))
            binding_site_com = md.compute_center_of_mass(aligned_traj.atom_slice(binding_site_indices))

            # Calculate distances between COMs
            for i in range(len(aligned_traj)):
                dist = np.linalg.norm(ligand_com[i] - binding_site_com[i])
                distances.append(dist)
        else:
            print("  WARNING: No binding site atoms found")

        # 3. Dihedral Analysis
        dihedral_data = []
        if len(ligand_indices) >= 4:
            example_dihedral = [ligand_indices[i] for i in range(4)]  # First 4 atoms as example
            dihedral_values = md.compute_dihedrals(aligned_traj, [example_dihedral])
            dihedral_data = dihedral_values.flatten()


        # Analysis functions
        def autocorr(x):
            """Calculate autocorrelation of time series"""
            result = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
            return result[result.size // 2:] / np.sum((x - np.mean(x)) ** 2)


        def get_dominant_period_fft(signal_data, time_per_frame=1.0):
            """Use FFT to find dominant period in a signal"""
            signal_detrended = signal_data - np.mean(signal_data)
            fft_result = np.abs(np.fft.rfft(signal_detrended))
            freqs = np.fft.rfftfreq(len(signal_data), d=time_per_frame)
            dominant_idx = np.argmax(fft_result[1:]) + 1
            dominant_freq = freqs[dominant_idx]
            return 1.0 / dominant_freq if dominant_freq != 0 else None


        def find_period_from_peaks(signal_data, height_thresh=0.1, time_per_frame=1.0):
            """Find period by detecting peaks in autocorrelation"""
            acf = autocorr(signal_data)
            peaks, _ = signal.find_peaks(acf, height=height_thresh)
            if len(peaks) >= 2:
                peak_diffs = np.diff(peaks)
                avg_period = np.mean(peak_diffs) * time_per_frame
                return avg_period
            else:
                return None


        # Analyze each time series
        time_per_frame = 1.0  # Adjust based on your simulation

        # Results for this specific analysis
        current_results = {
            'ligand': lig,
            'replicate': rep,
            'data_type': [],
            'period_peaks': [],
            'period_fft': []
        }

        # Analyze RMSD
        if len(rmsd_data) > 0:
            current_results['data_type'].append('RMSD')
            current_results['period_peaks'].append(find_period_from_peaks(rmsd_data, time_per_frame=time_per_frame))
            current_results['period_fft'].append(get_dominant_period_fft(rmsd_data, time_per_frame=time_per_frame))

        # Analyze distances
        if len(distances) > 0:
            current_results['data_type'].append('COM Distance')
            current_results['period_peaks'].append(find_period_from_peaks(distances, time_per_frame=time_per_frame))
            current_results['period_fft'].append(get_dominant_period_fft(distances, time_per_frame=time_per_frame))

        # Analyze dihedrals
        if len(dihedral_data) > 0:
            current_results['data_type'].append('Dihedral')
            current_results['period_peaks'].append(find_period_from_peaks(dihedral_data, time_per_frame=time_per_frame))
            current_results['period_fft'].append(get_dominant_period_fft(dihedral_data, time_per_frame=time_per_frame))

        # Create results table
        results_df = pd.DataFrame({
            'data_type': current_results['data_type'],
            'period_peaks': current_results['period_peaks'],
            'period_fft': current_results['period_fft']
        })

        print(f"  Results for {lig} replicate {rep}:")
        print(results_df)

        # Add to overall results
        for i, data_type in enumerate(current_results['data_type']):
            all_results.append({
                'ligand': lig,
                'replicate': rep,
                'data_type': data_type,
                'period_peaks': current_results['period_peaks'][i],
                'period_fft': current_results['period_fft'][i]
            })

        # Plot time series and their analyses
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"Oscillation Analysis: {lig} Replicate {rep}", fontsize=16)
        plot_index = 1

        # Plot RMSD
        if len(rmsd_data) > 0:
            # Time series
            plt.subplot(3, 3, plot_index)
            plt.plot(rmsd_data)
            plt.title(f'{lig} Rep{rep} RMSD vs Time')
            plt.xlabel('Frame')
            plt.ylabel('RMSD (nm)')
            plot_index += 1

            # Autocorrelation
            plt.subplot(3, 3, plot_index)
            acf_rmsd = autocorr(rmsd_data)
            plt.plot(acf_rmsd)
            if current_results['period_peaks'][0]:
                period_frames = int(current_results['period_peaks'][0] / time_per_frame)
                plt.axvline(x=period_frames, color='r', linestyle='--')
            plt.title(f'{lig} Rep{rep} RMSD Autocorrelation')
            plt.xlabel('Lag (frames)')
            plot_index += 1

            # Power spectrum
            plt.subplot(3, 3, plot_index)
            signal_fft = np.abs(np.fft.rfft(rmsd_data - np.mean(rmsd_data)))
            freqs = np.fft.rfftfreq(len(rmsd_data), d=time_per_frame)
            plt.plot(freqs[1:], signal_fft[1:])  # Skip zero frequency
            plt.title(f'{lig} Rep{rep} RMSD Power Spectrum')
            plt.xlabel('Frequency (1/frame)')
            plt.ylabel('Power')
            plot_index += 1

        # Plot distances
        if len(distances) > 0:
            # Time series
            plt.subplot(3, 3, plot_index)
            plt.plot(distances)
            plt.title(f'{lig} Rep{rep} COM Distance vs Time')
            plt.xlabel('Frame')
            plt.ylabel('Distance (nm)')
            plot_index += 1

            # Autocorrelation
            plt.subplot(3, 3, plot_index)
            acf_dist = autocorr(np.array(distances))
            plt.plot(acf_dist)
            if len(current_results['period_peaks']) > 1 and current_results['period_peaks'][1]:
                period_frames = int(current_results['period_peaks'][1] / time_per_frame)
                plt.axvline(x=period_frames, color='r', linestyle='--')
            plt.title(f'{lig} Rep{rep} Distance Autocorrelation')
            plt.xlabel('Lag (frames)')
            plot_index += 1

            # Power spectrum
            plt.subplot(3, 3, plot_index)
            signal_fft = np.abs(np.fft.rfft(np.array(distances) - np.mean(distances)))
            freqs = np.fft.rfftfreq(len(distances), d=time_per_frame)
            plt.plot(freqs[1:], signal_fft[1:])  # Skip zero frequency
            plt.title(f'{lig} Rep{rep} Distance Power Spectrum')
            plt.xlabel('Frequency (1/frame)')
            plt.ylabel('Power')
            plot_index += 1

        # Plot dihedral
        if len(dihedral_data) > 0:
            # Time series
            plt.subplot(3, 3, plot_index)
            plt.plot(dihedral_data)
            plt.title(f'{lig} Rep{rep} Dihedral Angle vs Time')
            plt.xlabel('Frame')
            plt.ylabel('Angle (radians)')
            plot_index += 1

            # Autocorrelation
            plt.subplot(3, 3, plot_index)
            acf_dih = autocorr(dihedral_data)
            plt.plot(acf_dih)
            if len(current_results['period_peaks']) > 2 and current_results['period_peaks'][2]:
                period_frames = int(current_results['period_peaks'][2] / time_per_frame)
                plt.axvline(x=period_frames, color='r', linestyle='--')
            plt.title(f'{lig} Rep{rep} Dihedral Autocorrelation')
            plt.xlabel('Lag (frames)')
            plot_index += 1

            # Power spectrum
            plt.subplot(3, 3, plot_index)
            signal_fft = np.abs(np.fft.rfft(dihedral_data - np.mean(dihedral_data)))
            freqs = np.fft.rfftfreq(len(dihedral_data), d=time_per_frame)
            plt.plot(freqs[1:], signal_fft[1:])  # Skip zero frequency
            plt.title(f'{lig} Rep{rep} Dihedral Power Spectrum')
            plt.xlabel('Frequency (1/frame)')
            plt.ylabel('Power')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        output_file = f"oscillation_analysis_{lig}_rep{rep}.png"
        plt.savefig(output_file)
        plt.close()  # Close figure to free memory
        print(f"  Saved analysis to {output_file}")

# After all analyses are complete, create a summary table with all results
if all_results:
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("oscillation_periods_summary.csv", index=False)
    print("Complete analysis summary saved to oscillation_periods_summary.csv")

    # Display summary statistics by ligand
    print("\nSummary of oscillation periods by ligand:")
    for lig in lig_list:
        lig_data = summary_df[summary_df['ligand'] == lig]
        if not lig_data.empty:
            print(f"\n{lig}:")
            # Group by data_type and calculate mean periods
            grouped = lig_data.groupby('data_type').agg({
                'period_peaks': ['mean', 'std'],
                'period_fft': ['mean', 'std']
            })
            print(grouped)

print("Analysis complete for all ligands and replicates.")