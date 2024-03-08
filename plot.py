from matplotlib import pyplot as plt

# Sample data (replace with your actual data)
x = [1, 2, 3, 4, 5]
y1 = [2, 5, 7, 3, 1]
y2 = [40, 80, 60, 20, 100]  # Percentage data (0 to 100)

# Create the figure and primary axes
fig, ax1 = plt.subplots(figsize=(4, 3))

# Plot the line with normal scale on primary y-axis
p1, = ax1.plot(x, y1, label='Normal Scale')

# Create a secondary y-axis for percentage data
ax2 = ax1.twinx()

# Plot the line with percentage on secondary y-axis (set limits from 0 to 100)
p2, = ax2.plot(x, y2, label='Percentage', color='red')
ax2.set_ylim(0, 100)  # Set limits for percentage axis

# Set labels and title
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Normal Scale')
ax2.set_ylabel('Percentage')
plt.title('Plot with Two Scales')

# Show legend
plt.legend(handles=[p1, p2])

# Adjust layout to avoid overlapping labels
plt.tight_layout()

plt.show()
# Plot new data directly on top (replace with your new data)
x_new = [6, 7, 8]
y1_new = [8, 2, 4]
y2_new = [30, 50, 70]

ax1.plot(x_new, y1_new, label='Normal Scale')  # Plot on primary axis
ax2.plot(x_new, y2_new, label='Percentage', color='red')  # Plot on secondary axis
ax2.set_ylim(0, 100)

# You can add more code here for further customization of the new plot

plt.show()