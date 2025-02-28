import stim
import pymatching
import logging
import numpy as np
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Starting MWPM decoding using Stim and PyMatching.")

# Generate a surface code circuit with built-in noise
distance = 3
rounds = 10
physical_error_rate = 0.01  # Increase to 0.005 if no errors appear

logging.info(f"Generating surface code circuit with distance={distance}, rounds={rounds}, and noise {physical_error_rate}.")
circuit = stim.Circuit.generated(
    code_task="surface_code:rotated_memory_x",
    distance=distance,
    rounds=rounds,
    after_clifford_depolarization=physical_error_rate,  # Inject noise directly
)

# Simulate detection events (syndrome extraction)
shots = 100
logging.info(f"Simulating syndrome extraction with {shots} shots.")
sampler = circuit.compile_detector_sampler()
syndrome_data = sampler.sample(shots=shots)

# Debug syndrome data
logging.info(f"Syndrome data sample:\n{syndrome_data[:5]}")
logging.info(f"Number of nonzero syndromes: {np.count_nonzero(syndrome_data)}")

# Convert Stim's detector graph into a PyMatching decoder
logging.info("Extracting detector error model from Stim circuit.")
detector_graph = circuit.detector_error_model()
matching_decoder = pymatching.Matching.from_detector_error_model(detector_graph)
logging.info(f"Detector graph summary: {detector_graph}")
logging.info(f"Number of detectors in the graph: {matching_decoder.num_detectors}")
logging.info(f"Number of edges in the graph: {len(list(matching_decoder.edges()))}")


matching_decoder.draw()
plt.show()

# Test decoding for a single syndrome shot
first_syndrome = syndrome_data[0]
logging.info(f"Example syndrome before decoding: {first_syndrome}")

decoded_correction = matching_decoder.decode(first_syndrome)
logging.info(f"Decoded correction for first shot (raw): {decoded_correction}")

# Print locations where the correction is nonzero
nonzero_corrections = [i for i, val in enumerate(decoded_correction) if val]
logging.info(f"Nonzero correction locations for first shot: {nonzero_corrections}")


# Decode using MWPM
logging.info("Decoding syndromes using MWPM.")
corrections = matching_decoder.decode_batch(syndrome_data)

nontrivial_corrections = sum(1 for c in corrections if any(c))
logging.info(f"Number of non-trivial corrections: {nontrivial_corrections}/{len(corrections)}")

# Print the first 5 decoded results
logging.info("Displaying the MWPM corrections:")
for i, correction in enumerate(corrections):
    logging.info(f"Shot {i+1}: {correction}")

for i, correction in enumerate(corrections):  # all shots
    if any(correction):
        logging.info(f"Shot {i+1} corrections: {correction}")


logging.info("MWPM decoding completed successfully.")
