import streamlit as st
from pi2_sphere_updated import Pi2Framework  # Import your updated framework

# Initialize framework (adjust params if needed)
fw = Pi2Framework()

st.title("Pi2 Sphere Reality Simulation")
zoom_level = st.slider("Zoom Level (-1 Vast to 1 Detailed)", -1.0, 1.0, 0.0)
t = st.slider("Time (years)", 0.0, 1e10, 0.0)
neutrality = st.slider("Law Neutrality (0=Neutral)", 0.0, 1.0, 0.0)
fw.law_neutrality_factor = neutrality
scale = st.selectbox("Initial Scale to Simulate", ["cosmic", "planetary", "human"])

# Generate or load entities for selected scale
if st.button("Generate Entities"):
    fw.cosmo_core.spatial_hierarchy.map_real_data(scale)  # Use real_data_sources[scale] if set
    st.success(f"Generated {len(fw.cosmo_core.spatial_hierarchy.entities.get(scale, []))} entities for {scale} scale.")

# Run simulation if desired
if st.button("Run Simulation (10 steps)"):
    final_t = fw.run_simulation(steps=10, t_start=t)
    st.success(f"Simulation advanced to t={final_t}")

# Render view
fig = fw.utils.visualizer.render_sphere_view(zoom_level, t, fw.neutral_observer_pos)
st.plotly_chart(fig, use_container_width=True)

# Sidebar metrics
st.sidebar.title("Metrics")
st.sidebar.write(f"Current Tension: {fw.cosmo_core.hybrid_de_tension(0.5, t=t):.4f}")
st.sidebar.write(f"Entity Count (Visible): {len(fw.cosmo_core.spatial_hierarchy.get_view(zoom_level, fw.neutral_observer_pos))}")