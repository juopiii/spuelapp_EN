import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Core Simulation Logic (remains mostly the same internally) ---
def run_simulation(L_segments, d_segments, V_extra_initial, Q, rho, mu, D_mol,
                   prob_enter_trap, prob_exit_trap, N_particles, dt,
                   contamination_threshold, max_sim_time_factor, status_placeholder):
    """Runs the particle simulation."""

    num_segments = len(L_segments)
    if num_segments == 0:
        status_placeholder.error("Error: No segments defined.")
        return None # Return None or raise error if no segments

    r_segments = [d / 2.0 for d in d_segments]
    A_segments = [np.pi * r**2 for r in r_segments]

    A_last = A_segments[-1] if A_segments else 1e-12 # Use last segment's area
    L_extra = V_extra_initial / A_last if A_last > 1e-12 else 0

    pos_end_segments = np.cumsum(L_segments)
    L_total_segments = pos_end_segments[-1]
    L_total_effective = L_total_segments + L_extra

    v_segments = [(Q / A) if A > 1e-12 else 0 for A in A_segments]
    Re_segments = [(rho * v * d / mu) if mu > 1e-12 else 0
                   for v, d in zip(v_segments, d_segments)]

    D_eff_segments = []
    is_turbulent = []
    for i in range(num_segments):
        Re = Re_segments[i]
        r = r_segments[i]
        v = v_segments[i]
        turbulent = False
        d_mol_safe = max(D_mol, 1e-15) # Prevent division by zero
        # Base Taylor-Aris
        D_eff = d_mol_safe + (r**2 * v**2) / (48 * d_mol_safe)

        if Re >= 4000: # Turbulent
            D_eff *= 10 # Rough approximation factor for turbulence
            turbulent = True
        elif Re >= 2000: # Transitional
            # Keep base Taylor-Aris but flag it
            turbulent = True # Flag as transitional/turbulent

        D_eff_segments.append(D_eff)
        is_turbulent.append(turbulent)

    V_segments_calc = [A * L for A, L in zip(A_segments, L_segments)]
    V_total_effective_calc = sum(V_segments_calc) + V_extra_initial
    t_displacement_effective = V_total_effective_calc / Q if Q > 1e-15 else 0

    # --- Simulation Initialization ---
    status_placeholder.text("Initializing simulation...")
    particle_pos = np.linspace(0, L_total_effective, N_particles)
    particle_is_old = np.ones(N_particles)
    particle_is_trapped = np.zeros(N_particles, dtype=bool)

    time_elapsed = 0.0
    volume_pumped = 0.0
    outlet_concentration_history = []
    time_history = []
    avg_last_concentration = 1.0

    start_sim_time_real = time.time()
    max_sim_time = t_displacement_effective * max_sim_time_factor if t_displacement_effective > 0 else 600.0 # Fallback max time

    loop_counter = 0
    update_interval = 200 # Update status less frequently for potentially longer runs

    while time_elapsed < max_sim_time:
        loop_counter += 1

        # Tailing Model
        if prob_exit_trap > 0:
            can_be_released_mask = particle_is_trapped
            if np.any(can_be_released_mask):
                exit_roll = np.random.rand(np.sum(can_be_released_mask)) < prob_exit_trap
                indices_to_release = np.where(can_be_released_mask)[0][exit_roll]
                if len(indices_to_release) > 0:
                    particle_is_trapped[indices_to_release] = False
        if prob_enter_trap > 0:
            can_be_trapped_mask = ~particle_is_trapped
            if np.any(can_be_trapped_mask):
                enter_roll = np.random.rand(np.sum(can_be_trapped_mask)) < prob_enter_trap
                indices_to_trap = np.where(can_be_trapped_mask)[0][enter_roll]
                if len(indices_to_trap) > 0:
                     particle_is_trapped[indices_to_trap] = True

        mobile_mask = ~particle_is_trapped

        # Movement only for mobile particles
        if np.any(mobile_mask):
            mobile_indices = np.where(mobile_mask)[0]
            current_pos = particle_pos[mobile_mask]

            conditions_v = []
            choices_v = []
            conditions_D = []
            choices_D = []
            last_pos = 0.0
            for i in range(num_segments):
                conditions_v.append((current_pos >= last_pos) & (current_pos < pos_end_segments[i]))
                choices_v.append(v_segments[i])
                conditions_D.append((current_pos >= last_pos) & (current_pos < pos_end_segments[i]))
                choices_D.append(D_eff_segments[i])
                last_pos = pos_end_segments[i]

            # Append conditions/choices for the extra volume part (using last segment's properties)
            conditions_v.append(current_pos >= last_pos)
            choices_v.append(v_segments[-1])
            conditions_D.append(current_pos >= last_pos)
            choices_D.append(D_eff_segments[-1])

            velocities = np.select(conditions_v, choices_v, default=choices_v[-1])
            D_effs = np.select(conditions_D, choices_D, default=choices_D[-1])

            particle_pos[mobile_mask] += velocities * dt

            safe_D_effs = np.maximum(D_effs, 1e-15)
            sqrt_term = np.sqrt(2.0 * safe_D_effs * dt)
            random_steps = np.random.standard_normal(len(mobile_indices)) * sqrt_term
            particle_pos[mobile_mask] += random_steps

        # Boundary Conditions
        exited_mask = particle_pos >= L_total_effective
        n_exited = np.sum(exited_mask)

        if n_exited > 0:
            old_exited_mask = exited_mask & (particle_is_old == 1)
            n_old_exited = np.sum(old_exited_mask)
            current_outlet_concentration = n_old_exited / n_exited
        else:
            current_outlet_concentration = outlet_concentration_history[-1] if outlet_concentration_history else 0.0

        outlet_concentration_history.append(current_outlet_concentration)
        time_history.append(time_elapsed)

        keep_mask = ~exited_mask
        particle_pos = particle_pos[keep_mask]
        particle_is_old = particle_is_old[keep_mask]
        particle_is_trapped = particle_is_trapped[keep_mask]

        # New Particles
        n_new = n_exited
        if n_new > 0:
            v_entry = v_segments[0] # Entry velocity from the first segment
            new_particles_pos = np.random.uniform(0, max(v_entry * dt, 1e-9), n_new)
            new_particles_old = np.zeros(n_new)
            new_particles_trapped = np.zeros(n_new, dtype=bool)
            particle_pos = np.concatenate((particle_pos, new_particles_pos))
            particle_is_old = np.concatenate((particle_is_old, new_particles_old))
            particle_is_trapped = np.concatenate((particle_is_trapped, new_particles_trapped))

        # Time/Volume Update
        time_elapsed += dt
        volume_pumped = Q * time_elapsed

        # Progress / Termination Check
        if loop_counter % update_interval == 0 or n_exited > 0:
             num_trapped = np.sum(particle_is_trapped)
             avg_window = min(len(outlet_concentration_history), 100)
             if avg_window > 0:
                 avg_last_concentration = np.mean(outlet_concentration_history[-avg_window:])
             else:
                 avg_last_concentration = 1.0

             status_placeholder.text(
                 f"Sim: {time_elapsed:.1f}s ({time_elapsed/60:.2f}m), "
                 f"Vol: {volume_pumped*1e6:.1f}mL, "
                 f"Avg Outlet Conc: {avg_last_concentration*100:.3f}%, "
                 f"Trapped: {num_trapped}"
             )

             if avg_last_concentration < contamination_threshold and time_elapsed > t_displacement_effective * 0.5 :
                  status_placeholder.text("Target contamination level reached!")
                  break

    # --- End of Simulation ---
    end_sim_time_real = time.time()
    sim_duration_real = end_sim_time_real - start_sim_time_real
    final_volume = volume_pumped
    final_time = time_elapsed

    status_message = "Simulation completed."
    if time_elapsed >= max_sim_time and avg_last_concentration >= contamination_threshold :
        status_message = f"WARNING: Max. sim time ({max_sim_time:.1f}s) reached BEFORE target contamination ({contamination_threshold*100:.1f}%). Last avg conc: {avg_last_concentration*100:.3f}%."

    return (final_time, final_volume, time_history, outlet_concentration_history,
            t_displacement_effective, V_total_effective_calc, avg_last_concentration,
            sim_duration_real, status_message,
            v_segments, Re_segments, D_eff_segments, is_turbulent, pos_end_segments)

# --- Streamlit UI (Translated and with Categories) ---
st.set_page_config(layout="wide")
st.title("🔬 Flushing Volume Simulation for Tubing Systems")

# --- Sidebar for Inputs ---
st.sidebar.header("System Configuration")

num_segments = st.sidebar.slider("Number of tube segments", 1, 5, 3)

L_segments_m = []
d_segments_mm = []
# Provide more distinct default values for clarity
default_lengths = [60.0, 50.0, 20.0, 10.0, 10.0]
default_diameters = [2.0, 4.0, 8.0, 8.0, 8.0]
for i in range(num_segments):
    st.sidebar.subheader(f"Segment {i+1}")
    # Use defaults based on index i
    l_cm = st.sidebar.number_input(f"Length Segment {i+1} (cm)", min_value=0.1, value=default_lengths[i], step=0.1, key=f"L_{i}")
    d_mm = st.sidebar.number_input(f"Inner Diameter Segment {i+1} (mm)", min_value=0.1, value=default_diameters[i], step=0.1, key=f"d_{i}")
    L_segments_m.append(l_cm / 100.0) # Convert to meters
    d_segments_mm.append(d_mm)

V_extra_initial_ml = st.sidebar.number_input("Additional initial volume at end (mL)", min_value=0.0, value=15.0, step=0.1)

st.sidebar.header("Flow and Fluid Properties")
Q_ml_per_min = st.sidebar.number_input("Flow Rate (mL/min)", min_value=0.1, value=200.0, step=1.0)
rho = st.sidebar.number_input("Density (kg/m³)", min_value=100.0, value=1000.0, step=10.0)
mu = st.sidebar.number_input("Dynamic Viscosity (Pa·s)", min_value=1e-5, value=1e-3, step=1e-4, format="%.4f")
D_mol = st.sidebar.number_input("Molecular Diffusion Coeff. (m²/s)", min_value=1e-12, value=1e-9, step=1e-10, format="%.2e")

st.sidebar.header("Tailing Model (Stagnation/Adsorption)")
st.sidebar.caption("Simulates slow release of residual contamination.")

# --- NEW: Tailing Categories ---
tailing_categories = {
    "1: Very Low":    (0.0001, 0.01),    # P(enter), P(exit) - Exit much easier
    "2: Low":         (0.001,  0.005),
    "3: Medium":      (0.005,  0.001),   # More balanced, noticeable effect
    "4: High":        (0.005,  0.0001),  # Harder to exit
    "5: Very High":   (0.01,   0.00005)  # Even harder to exit, higher entry prob
}
selected_category = st.sidebar.selectbox(
    "Contamination Persistence Level:",
    options=list(tailing_categories.keys()),
    index=3 # Default to "High"
)
# Get the probability pair for the selected category
prob_enter_trap, prob_exit_trap = tailing_categories[selected_category]
st.sidebar.caption(f"Using: P(Enter)={prob_enter_trap:.5f}, P(Exit)={prob_exit_trap:.6f}")
# Removed the direct number inputs for probabilities


# --- Expander for Simulation Settings ---
with st.sidebar.expander("Simulation Parameters (Advanced)"):
    N_particles = st.number_input("Number of Particles", min_value=1000, value=20000, step=1000)
    dt = st.number_input("Time Step dt (s)", min_value=1e-4, value=0.005, step=1e-3, format="%.4f", help="Smaller dt might be needed for very high velocities or dispersions, but increases simulation time.")
    contamination_threshold_perc = st.number_input("Target Residual Contamination (%)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    contamination_threshold = contamination_threshold_perc / 100.0 # Convert to fraction
    max_sim_time_factor = st.number_input("Max. Simulation Time (x theoretical displacement)", min_value=1, value=25, step=1)

# --- Main Area for Output ---
st.header("Simulation Run")

if st.button("▶️ Start Simulation"):
    # Convert diameter from mm to m
    d_segments_m = [d/1000.0 for d in d_segments_mm]
    # Convert flow rate to m³/s
    Q_m3_s = Q_ml_per_min * 1e-6 / 60.0
    # Convert extra volume to m³
    V_extra_initial_m3 = V_extra_initial_ml * 1e-6

    # Placeholders for status and figure
    status_placeholder = st.empty()
    fig_placeholder = st.empty()

    with st.spinner("Running simulation... Please wait."):
        try:
            results = run_simulation(
                L_segments=L_segments_m,
                d_segments=d_segments_m,
                V_extra_initial=V_extra_initial_m3,
                Q=Q_m3_s,
                rho=rho,
                mu=mu,
                D_mol=D_mol,
                prob_enter_trap=prob_enter_trap, # Use selected category values
                prob_exit_trap=prob_exit_trap,   # Use selected category values
                N_particles=N_particles,
                dt=dt,
                contamination_threshold=contamination_threshold,
                max_sim_time_factor=max_sim_time_factor,
                status_placeholder=status_placeholder
            )

            if results: # Check if simulation returned results
                (final_time, final_volume, time_history, outlet_concentration_history,
                 t_displacement_effective, V_total_effective_calc, avg_last_concentration,
                 sim_duration_real, status_message,
                 v_segments, Re_segments, D_eff_segments, is_turbulent, pos_end_segments) = results

                st.success(f"Simulation finished in {sim_duration_real:.2f} seconds (real time).")
                st.info(status_message)

                # --- Show Results ---
                st.subheader("Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Required Flush Time", f"{final_time:.1f} s", f"{final_time/60:.2f} min")
                col2.metric("Required Flush Volume", f"{final_volume * 1e6:.2f} mL")
                col3.metric("Flush Volume Factor", f"{final_volume / V_total_effective_calc:.1f} x" if V_total_effective_calc > 1e-12 else "N/A", f"Total Vol: {V_total_effective_calc * 1e6:.2f} mL")

                # --- Show Calculated Parameters ---
                with st.expander("Show Calculated System Parameters"):
                     st.markdown(f"**Theoretical Displacement Time:** {t_displacement_effective:.1f} s ({t_displacement_effective/60:.2f} min)")
                     st.markdown("**Segment Details:**")
                     param_data = {
                         "Segment": [f"{i+1}" for i in range(num_segments)],
                         "Length (m)": [f"{l:.3f}" for l in L_segments_m],
                         "Diameter (mm)": d_segments_mm,
                         "Velocity v (m/s)": [f"{v:.3f}" for v in v_segments],
                         "Re": [f"{Re:.1f}" for Re in Re_segments],
                         "D_eff (m²/s)": [f"{D:.2e}" for D in D_eff_segments],
                         "Turbulent/Trans?": ["Yes" if turb else "No" for turb in is_turbulent]
                     }
                     st.dataframe(param_data)


                # --- Create Plot ---
                st.subheader("Flushing Curve")
                fig, ax = plt.subplots(figsize=(10, 5)) # Create new figure

                time_array_min = np.array(time_history) / 60.0
                conc_array_perc = np.array(outlet_concentration_history) * 100.0

                # Smoothing for plot
                smooth_window = 20
                if len(time_history) > smooth_window:
                    smooth_conc = np.convolve(conc_array_perc, np.ones(smooth_window)/smooth_window, mode='valid')
                    valid_indices = len(time_array_min) - len(smooth_conc)
                    start_index = valid_indices // 2
                    end_index = start_index + len(smooth_conc)
                    smooth_time_min = time_array_min[start_index:end_index]
                    if len(smooth_time_min)>0:
                         ax.plot(smooth_time_min, smooth_conc, label='Outlet Conc. (Smoothed)', zorder=5)
                    else:
                         ax.plot(time_array_min, conc_array_perc, label='Outlet Conc. (Raw)', zorder=5) # Fallback
                else:
                     ax.plot(time_array_min, conc_array_perc, label='Outlet Conc. (Raw)', zorder=5)

                ax.axhline(contamination_threshold * 100, color='r', linestyle='--', label=f'Target: {contamination_threshold*100:.2f}%', zorder=1)
                if t_displacement_effective > 0 :
                    ax.axvline(t_displacement_effective / 60, color='g', linestyle=':', label=f'Theor. Displ. Time ({t_displacement_effective:.1f}s)', zorder=1)
                if final_time > 0:
                    ax.axvline(final_time / 60, color='k', linestyle='-', label=f'Sim. End ({final_time/60:.1f} min)', zorder=1)

                # Experimental reference line (example: 100mL)
                exp_ref_vol = 100.0
                exp_ref_time_min = exp_ref_vol / Q_ml_per_min if Q_ml_per_min > 0 else 0
                if exp_ref_time_min > 0:
                     ax.axvline(exp_ref_time_min, color='orange', linestyle='-.', label=f'Ref: {exp_ref_vol:.0f}mL Limit', zorder = 1)


                ax.set_xlabel("Time (minutes)")
                ax.set_ylabel("Old Sample Concentration at Outlet (%)")
                ax.set_title(f"Simulated Flushing Curve (Q={Q_ml_per_min:.0f} mL/min)")
                ax.legend(fontsize='small')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                # Dynamic axis limits
                top_lim = 5 if len(conc_array_perc) == 0 else max(5, np.max(conc_array_perc) * 1.1)
                ax.set_ylim(bottom=-0.5, top=top_lim)
                plot_end_time_min = 0.1 # Minimum plot width
                if final_time > 0 :
                    plot_end_time_min = max(final_time / 60 * 1.1, plot_end_time_min)
                if exp_ref_time_min > 0:
                     plot_end_time_min = max(plot_end_time_min, exp_ref_time_min * 1.1)
                ax.set_xlim(left=min(0, -0.05 * plot_end_time_min), right=plot_end_time_min)

                # Optional Log Scale (can be added with a checkbox later)
                # y_log_scale = st.checkbox("Use Logarithmic Y-Axis for Concentration?")
                # if y_log_scale:
                #    ax.set_yscale('log')
                #    ax.set_ylim(bottom=max(contamination_threshold*100 * 0.1, 0.01)) # Adjust lower limit for log

                plt.tight_layout()
                fig_placeholder.pyplot(fig) # Display plot in placeholder
                plt.close(fig) # Close figure to free memory

        except Exception as e:
            st.error(f"An error occurred during the simulation: {e}")
            import traceback
            st.exception(e) # More detailed traceback in Streamlit

else:
    st.info("Configure the system in the sidebar and click 'Start Simulation'.")