#### Model Training Trajectory Plotting ####
## Prepared by: Sneha Kumar
import matplotlib.pyplot as plt
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('dbert_base_model_3_complete.pt', map_location=device)      #replace with model checkpoint of the model trajectory to be plotted

plt.style.use('ggplot')

# Create a range of x values (assuming the lists have sequential data)
x = range(len(checkpoint['train_losses']))

plt.figure(figsize=(20, 4))

# Plot the trajectory of training
plt.plot(x, checkpoint['train_losses'], label='Training BCE Loss')
plt.plot(x, checkpoint['train_f1_scores'], label= 'Training F1-Score')
print('plotted train')

# Plot the trajectory of validation
valid_x = [round(checkpoint['step']/10) * i for i in range(1, checkpoint['epoch'] + 1)]
plt.plot(valid_x, checkpoint['valid_losses'], label = "Validation BCE Loss")
plt.plot(valid_x, checkpoint['valid_f1_scores'], label = "Validation F1-Scores")
print('plotted valid')

# Add labels and legend
plt.xlabel('Every 10th Step')
plt.ylabel('Value')
plt.legend()

# Show the plot
#plt.show()
plt.savefig('Evaluations_and_Plots/trajectory_plot.png')
