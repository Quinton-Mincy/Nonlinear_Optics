import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def process_file(filename):
    """Read and process a single file to extract energies and intensities."""
    energies = []
    intensities = []
    integration_time = None
    convert = False
    with open(filename, 'r') as file:
        data_section = False
        for line in file:
            line = line.strip()
            if line.startswith('#IntegrationTime'):
                integration_time = float(line.split(';')[1])
            if line.startswith('#XAxisUnit;eV'):
                convert = True
            if line == '[Data]':
                data_section = True
                continue
            if data_section:
                if line == '[EndOfFile]':
                    break
                energy, intensity = map(float, line.split(';'))
                energies.append(energy)
                intensities.append(intensity)
    return np.array(energies), (np.array(intensities) / (integration_time / 1000)), convert

def save_processed_data(output_path, measurement_num, wavelengths, intensities):
    """Save the processed data to a CSV file."""
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f'Measurement_{measurement_num}.csv')
    data = np.column_stack((wavelengths, intensities))
    np.savetxt(output_file, data, delimiter=',', header='Wavelength,Intensity', comments='')
    print(f'Saved: {output_file}')

def process_sample(sample_path,sample_label):
    """Process all measurement and reference files in a given sample directory."""
    all_wavelengths = []
    all_intensities = []

    all_wavelengths_ref = []
    all_intensities_ref = []

    # Read all CSV files matching the pattern in the specified directory
    measurement_files = glob.glob(os.path.join(sample_path, 'MeasureAlq3_*.csv'))
    reference_files = glob.glob(os.path.join(sample_path, 'ReferenceGlass_*.csv'))
    print(os.path.join(sample_path, 'MeasureAlq3_*.csv'))
    # Ensure each measurement file has a corresponding reference file
    if len(measurement_files) != len(reference_files):
        raise ValueError("The number of measurement files does not match the number of reference files.")

    #extract data from files
    for measure_file, ref_file in zip(measurement_files, reference_files):
        measure_wavelengths, measure_intensities, convert = process_file(measure_file)
        ref_wavelengths, ref_intensities, con = process_file(ref_file)

        if convert:
            measure_wavelengths = 1239.513 / measure_wavelengths
            ref_wavelengths     = 1239.513 / ref_wavelengths

        all_wavelengths.append(measure_wavelengths)
        all_intensities.append(measure_intensities)

        all_wavelengths_ref.append(ref_wavelengths)
        all_intensities_ref.append(ref_intensities)

    # Ensure all wavelength arrays are the same
    if not all(np.array_equal(all_wavelengths[0], e) for e in all_wavelengths):
        raise ValueError("Wavelength values differ between files.")

    return all_wavelengths, all_intensities, all_wavelengths_ref, all_intensities_ref

def plot_data(ax, all_wavelengths, all_intensities, sample_label):
    """Plot the data from all files and their average on a given Axes object."""

    output_path = f".\csv_dir\{sample_label}"
    # Plot each file's data in different colors
    for i, (wavelengths, intensities) in enumerate(zip(all_wavelengths, all_intensities)):
        # Save the processed data to a CSV file
        ax.plot(wavelengths, intensities, label=f'{sample_label} Measurement {i+1}')
        save_processed_data(output_path, i + 1, wavelengths, intensities)


    # Calculate the average intensities
    average_intensities = np.mean(all_intensities, axis=0)
    
    # Plot the average intensity trace
    ax.plot(all_wavelengths[0], average_intensities, label=f'{sample_label} Average', color='black', linewidth=2, linestyle='--')
    # Save to file
    save_processed_data(output_path, "average", all_wavelengths[0], average_intensities)

    # Plot settings
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_title(f'Wavelength vs. Intensity - {sample_label}')
    ax.legend()
    ax.grid(True)

def main():
    sample1_path = 'transmission/Thinfilm_Interference/sample1'
    sample2_path = 'transmission/Thinfilm_Interference/sample2'
    sample3_path = 'transmission/Thinfilm_Interference/sample3'
    sample4_path = 'transmission/Thinfilm_Interference/sample4'
    sample5_path = 'transmission/Thinfilm_Interference/sample5'
    
    print('glob finds',glob.glob(sample1_path+'/*.csv'))
    # Create a figure with subplots
    fig, axs = plt.subplots(5, 1, figsize=(8, 8))
    paths = [sample1_path, sample2_path,sample3_path, sample4_path,sample5_path]
    labels= ['Sample_1','Sample_2','Sample_3','Sample_4', 'Sample_5']
    # Process each sample and plot the data
    for ax, sample_path, sample_label in zip(axs, paths, labels):

        all_wavelengths, all_intensities, all_wavelengths_ref, all_intensities_ref = process_sample(sample_path,sample_label)
        #divide out average reference intensity
        average_intensities = np.mean(all_intensities_ref, axis=0)
     
        #divide sample measurement intensities by avg reference intensity
        all_intensities = all_intensities/average_intensities

        plot_data(ax, all_wavelengths, all_intensities, sample_label)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
