import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

FILE_PATH_1 = "./lena.png"
FILE_PATH_2 = "./mountains.jpg"
FILE_PATH_3 = "./sofa.png"

def create_histogram(data_array, title, num_bins=256):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Create the histogram
    ax.hist(data_array, bins=num_bins, color='#66B2FF', edgecolor='black')

    # Add labels and title
    ax.set_xlabel('Valoare', fontsize=14)
    ax.set_ylabel('Frecventa', fontsize=14)
    ax.set_title(title, fontsize=16)
    
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12)
    
def create_lorenz_graph(xs_decrypted, ys_decrypted, zs_decrypted, xs_encrypted = [], ys_encrypted =[] , zs_encrypted =[]):
    global ani
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Define a function to update the plot with each frame
    def update(frame):
        ax.clear()
        ax.plot3D(xs_decrypted[:frame], ys_decrypted[:frame], zs_decrypted[:frame], lw=0.5, color='blue', label='Decrypted')
        if len(xs_encrypted) > 0:
            ax.plot3D(xs_encrypted[:frame], ys_encrypted[:frame], zs_encrypted[:frame], lw=0.5, color='red', label='Encrypted')
        ax.set_xlabel("axa X")
        ax.set_ylabel("axa Y")
        ax.set_zlabel("axa Z")
        ax.set_title("Atractor Lorenz")
        ax.legend()

    # Create the animation
    num_frames = min(len(xs_decrypted), len(xs_encrypted))
    ani = FuncAnimation(fig, update, frames=num_frames, interval=20)

# Perform image encryption using Lorenz chaotic system
def lorenz_chaos_encryption(image_array):
    # Generate chaotic random numbers
    chaotic_numbers = generate_chaotic_numbers_lorenz(image_array.shape[0] * image_array.shape[1] * 3 * 8, False)

    # Convert image array to binary vector
    binary_vector = np.array(np.unpackbits(image_array), dtype=int)
    
    # Generate encryption key based on chaotic numbers
    encryption_key = generate_encryption_key(chaotic_numbers)

    # Perform bitwise XOR encryption
    encrypted_vector = np.bitwise_xor(binary_vector, encryption_key).astype(np.uint8)
    encrypted_array = np.packbits(encrypted_vector).reshape(image_array.shape)
    
    return encrypted_array

# Function to generate encryption key based on chaotic numbers
def generate_encryption_key(chaotic_numbers): 
    avg_value = np.mean(chaotic_numbers)

    # Generate encryption key based on binarization rule
    encryption_key = np.where(chaotic_numbers < avg_value, 0, 1)
        
    return encryption_key.astype(np.uint8)

# Function to generate chaotic random numbers
def generate_chaotic_numbers_lorenz(size, is_decryption):
    global xs_for_encryption, ys_for_encryption, zs_for_encryption
    # Generate chaotic numbers using Lorenz system
    chaotic_numbers = np.zeros(size)
    dt = 1000
    x, y, z = 0.1, 0.1, 0.1
    sigma = 10
    rho = 28
    beta = 8/3
    if is_decryption and alter_sigma.get():
        sigma = 10 + 10**(-15.052)
    xs, ys, zs = np.empty(size // dt + 2), np.empty(size // dt + 2), np.empty(size // dt + 2)
    xs[0], ys[0], zs[0] = (0.1, 0.1, 0.1)
    for i in range(size):
        x_dot = sigma * (y - x)
        y_dot = x * (rho - z) - y
        z_dot = x * y - beta * z
        x += x_dot * 0.01 % 256
        y += y_dot * 0.01 % 256
        z += z_dot * 0.01 % 256
        
        # Compute data to print the Lorenz attractor
        if i % dt == 0 and generate_animation.get():
            index = i // dt
            xs_dot = sigma * (ys[index] - xs[index])
            ys_dot = xs[index] * (rho - zs[index]) - ys[index]
            zs_dot = xs[index] * ys[index] - beta * zs[index]
            xs[index + 1] = xs[index] + xs_dot * 0.01
            ys[index + 1] = ys[index] + ys_dot * 0.01
            zs[index + 1] = zs[index] + zs_dot * 0.01

        chaotic_numbers[i] = (x + y + z) % 256

    if not is_decryption:
        xs_for_encryption = xs.copy()
        ys_for_encryption = ys.copy()
        zs_for_encryption = zs.copy()
        
    if is_decryption and generate_animation.get():
        create_lorenz_graph(xs, ys, zs, xs_for_encryption, ys_for_encryption, zs_for_encryption)
    return chaotic_numbers.astype(np.uint8)

def lorenz_chaos_decryption(encrypted_array):
    # Regenerate the same chaotic numbers used during encryption 
    chaotic_numbers = generate_chaotic_numbers_lorenz(encrypted_array.shape[0] * encrypted_array.shape[1] * 3 * 8, True)

    # Convert encrypted array to binary vector
    encrypted_vector = np.array(np.unpackbits(encrypted_array), dtype=int)
    
    # Generate decryption key based on the same chaotic numbers and key method used for encryption
    decryption_key = generate_encryption_key(chaotic_numbers)
    
    # Perform bitwise XOR decryption
    decrypted_vector = np.bitwise_xor(encrypted_vector, decryption_key).astype(np.uint8)

    # Convert decrypted vector back to image array
    decrypted_array = np.packbits(decrypted_vector).reshape(encrypted_array.shape)
        
    return decrypted_array

# Function to handle button click event
def button_click(file_path):
    # Open the image file
    image = Image.open(file_path)

    # Convert image to RGB mode
    image = image.convert("RGB")

    # Convert to gray image
    image_gray = image.convert("L")

    # Convert the image to numpy arrays
    image_array = np.array(image)
    image_gray_array = np.array(image_gray)

    # Generate the histogram of the original gray image    
    create_histogram(image_gray_array.flatten(), "Histograma imaginii originale")

    # Start the chronometer
    start_time = time.time()
    # Perform image encryption using Lorenz chaotic system
    encrypted_array = lorenz_chaos_encryption(image_array)
    
    # Stop the chronometer
    end_time = time.time()
    
    # Perform image decryption using Lorenz chaotic system
    decrypted_array = lorenz_chaos_decryption(encrypted_array)
    
    # Convert encrypted array back to image
    encrypted_image = Image.fromarray(encrypted_array.astype(np.uint8))
    decrypted_image = Image.fromarray(decrypted_array.astype(np.uint8))
    
    # Compute the execution time of the encryption
    chronometer.config(text="Execution time: " + str("{:.3f}".format(end_time - start_time)) + " seconds")
    
    # Compute the correlation coefficient between the original and the encrypted image
    correlation_coefficient__encrypted = np.corrcoef(image_array.flatten(), encrypted_array.flatten())[0, 1]
    corr_coef_encrypted.config(text="Correlation Coefficient against encrypted: " + str("{:.5f}".format(correlation_coefficient__encrypted)))
    
    # Compute the correlation coefficient between the original and the decrypted image
    correlation_coefficient_decrypted = np.corrcoef(image_array.flatten(), decrypted_array.flatten())[0, 1]
    corr_coef_decrypted.config(text="Correlation Coefficient against decrypted: " + str("{:.5f}".format(correlation_coefficient_decrypted)))
    
    # Compute MSE - mean squared error against the encrypted image
    squared_errors_encrypted = (image_array - encrypted_array) ** 2
    mse_encrypted = np.mean(squared_errors_encrypted)
    mean_squared_error_encrypted.config(text="Mean Squared Error against encrypted: " + str("{:.3f}".format(mse_encrypted)))
    
    # Compute MSE - mean squared error against the decrypted image
    squared_errors_decrypted = (image_array - decrypted_array) ** 2
    mse_decrypted = np.mean(squared_errors_decrypted)
    mean_squared_error_decrypted.config(text="Mean Squared Error against decrypted: " + str("{:.3f}".format(mse_decrypted)))
    
    # Compute EC - Encryption Complexity / EQ - Encryption Quality
    histogram = np.array(image_gray.histogram())
    histogram_l = np.array(encrypted_image.convert("L").histogram())
    ec = (np.sum(np.abs(histogram - histogram_l)))/256
    eq_evaluation.config(text="Encryption Quality: " + str("{:.3f}".format(ec)))
    
    # Generate the histogram of the encrypted image    
    create_histogram(encrypted_array.flatten(), "Histograma imaginii criptate")

    # Convert encrypted image to Tkinter format
    encrypted_tk_image = ImageTk.PhotoImage(encrypted_image)
    decrypted_tk_image = ImageTk.PhotoImage(decrypted_image)
    
    # Display the encrypted image in a new window
    encrypted_window = tk.Toplevel(root)
    encrypted_window.title("Encrypted Image")
    encrypted_label = tk.Label(encrypted_window, image=encrypted_tk_image)
    encrypted_label.pack()
    
    # Display the decrypted image in a new window
    decrypted_window = tk.Toplevel(root)
    decrypted_window.title("Decrypted Image")
    decrypted_label = tk.Label(decrypted_window, image=decrypted_tk_image)
    decrypted_label.pack()

    # Keep a reference to the image to prevent it from being garbage collected
    decrypted_label.image = decrypted_tk_image
    encrypted_label.image = encrypted_tk_image
    
    # Show the histogram
    plt.show()
        
def main():
    global root, chronometer, corr_coef_encrypted, corr_coef_decrypted, mean_squared_error_decrypted, mean_squared_error_encrypted, eq_evaluation, generate_animation, alter_sigma
    root = tk.Tk()
    root.title("Image Encryption")
    
    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Set the window size and position
    window_width = 400
    window_height = 500
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2

    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
    
    alter_sigma = tk.BooleanVar()
    checkbox_alter_sigma = tk.Checkbutton(root, text="Alter sigma during decryption", variable=alter_sigma)
    checkbox_alter_sigma.place(relx=0.27, rely= 0.05)
        
    generate_animation = tk.BooleanVar()
    checkbox_animation = tk.Checkbutton(root, text="Generate Lorenz animation", variable=generate_animation)
    checkbox_animation.place(relx=0.27, rely= 0.1)
    
    # Create button 1
    button = tk.Button(root, text="lena.png", command=lambda: button_click(FILE_PATH_1))
    button.place(relx=0.5, rely=0.2, anchor=tk.CENTER)
    
    # Create button 2
    button = tk.Button(root, text="mountains.jpg", command=lambda: button_click(FILE_PATH_2))
    button.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
    
    # Create button 3
    button = tk.Button(root, text="sofa.png", command=lambda: button_click(FILE_PATH_3))
    button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
    
    evaluation_title = tk.Label(root, text="Evaluation measurement", font=("Helvetica", 12,"bold underline"))
    evaluation_title.place(relx=0.5, rely=0.55, anchor=tk.CENTER)
    
    chronometer = tk.Label(root, text="Execution time: 0 seconds")
    chronometer.place(relx = 0.5, rely=0.62, anchor = tk.CENTER)
    
    corr_coef_encrypted = tk.Label(root, text="Correlation Coefficient against encrypted: 0")
    corr_coef_encrypted.place(relx=0.5, rely=0.67, anchor=tk.CENTER)
    
    corr_coef_decrypted = tk.Label(root, text="Correlation Coefficient against decrypted: 0")
    corr_coef_decrypted.place(relx=0.5, rely=0.72, anchor=tk.CENTER)
    
    mean_squared_error_encrypted = tk.Label(root, text="Mean Squared Error against encrypted: 0")
    mean_squared_error_encrypted.place(relx=0.5, rely=0.77, anchor=tk.CENTER)
    
    mean_squared_error_decrypted = tk.Label(root, text="Mean Squared Error against decrypted: 0")
    mean_squared_error_decrypted.place(relx=0.5, rely=0.82, anchor=tk.CENTER)
    
    eq_evaluation = tk.Label(root, text="Encryption Quality: 0")
    eq_evaluation.place(relx=0.5, rely=0.87, anchor=tk.CENTER)
    root.mainloop()

if __name__ == "__main__":
    main()