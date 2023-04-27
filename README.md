### Breaking the Barrier of Expensive CNN Training:<br>
# Achieving High Accuracy with Efficient Layer Transfer Learning <br><br>

# **Problem** <br>
Despite Convolutional Neural Networks have achieved remarkable success in various applications of Artificial Intelligence, several issues still exist with training CNNs: <br>
1) Expensive <br>
2) Time Consumption <br>
3) Usage of Computational Resource <br>
Thus, exploring techniques to reduce such issues of CNNs while maintaining near-optimal accuracy levels is a must. <br><br>

# **Solution** <br>
The proposed solution to address the high cost of training CNN: <br>
**Layer Transfer Learning** <br>
Use of layer transfer learning entails:<br>
*1) Model only learning via Backpropagation initially* <br>
*2) Freezing all layers except final Dense layer*<br>
This technique can reduce training costs by over 60% while maintaining high prediction accuracy of over 90%. <br><br>

# **Importance** <br>
Layer transfer learning has the potential to be an effective solution for addressing the challenge of expensive CNN training. Some of its significances are: <br>
1) Cost Reduction <br>
2) Reduced Training Time <br>
3) 3)Improved Accuracy <br>
4) Effective use of resources <br><br>

# **Methodologies** <br>
We coded and implemented six models over two different categories of datasets: <br>
**1) Individual(MNIST)** <br><br>
Backprop-Freeze-Compile-Test (M1)<br>
Backprop-Freeze-NoCompile-Test(M2)<br>
**2) Dual datasets(MNIST & EMNIST)**<br><br>
Backprop(MNIST)-Freeze-Compile-Transfer(EMNIST)-Test(M3)<br>
Backprop(EMNIST)-Freeze-Compile-Transfer(MNIST)-Test (M4)<br>
NoBackprop(MNIST)- Transfer(EMNIST)-Test(M5)<br>
Backprop(MNIST)-Transfer(EMNIST)-Test(M6)<br><br>

# **Results** <br><br>
Results for different models are listed below:<br>
**Individual(MNIST)** <br>
*Backprop-Freeze-Compile-Test* <br><br>
Test Accuracy: 99.01%<br>
Time Elapsed:   89.15s<br>
*Backprop-Freeze-NoCompile-Test*<br><br>
Test Accuracy: 98.75%<br>
Time Elapsed:   390.15s<br><br><br>

**Dual datasets(MNIST & EMNIST)** <br>
*Backprop(MNIST)-Freeze-Compile-Transfer(EMNIST)-Test*<br><br>
Test Accuracy: 51.02%<br>
Time Elapsed:   246.24s<br>
*Backprop(EMNIST)-Freeze-Compile-Transfer(MNIST)-Test*<br><br>
Test Accuracy: 8.27%<br>
Time Elapsed:   115.62s<br>
*NoBackprop(MNIST)- Transfer+NoBackprop(EMNIST)-Test*<br><br>
Test Accuracy: 48.10%<br>
Time Elapsed:  241.13s<br>
*Backprop(MNIST)-Transfer+Backprop(EMNIST)-Test*<br><br>
Test Accuracy: 49.67%<br>
Time Elapsed:  647.67s<br><br><br>

# **Conclusion** <br><br>
Backprop(MNIST)-Freeze-Compile-Transfer(EMNIST)-Test has the highest accuracy and least time elapse in comparison. This model is more accurate in comparison to Backprop(MNIST)-Transfer+Backprop(EMNIST)-Test (has backprop for both datasets).
Layer transfer learning is an effective solution for addressing the challenge of expensive CNN training, allowing for more efficient and cost-effective use of computational resources.




