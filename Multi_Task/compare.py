#### Compare Sentiment vs Sentiment with Sarcasm ####
## Prepared by: Sneha Kumar

import matplotlib.pyplot as plt 
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
sent_only = torch.load('senti-task_complete_3.pt')
multi = torch.load('multi-task_complete_3.pt') 

plt.style.use('ggplot')

# Create a range of x values (assuming the lists have sequential data)
x = range(len(sent_only['valid_sent_f1_scores']))

plt.figure(figsize = (15, 6))

# Plot the comparison of F1 score trajectory 
plt.plot(x, sent_only['valid_sent_f1_scores'], label = 'Sentiment Only')
plt.plot(x, multi['valid_sent_f1_scores'], label = 'Sentiment with Sarcasm')

# Add labels and legend
plt.xlabel('Every 49th Step')
plt.ylabel('Validation F1 Score')
plt.legend()

plt.savefig('sent_vs_sarc.png')
