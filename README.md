# decentralizedSenseAndSend

##### Implementation of the work in https://ieeexplore.ieee.org/document/8494742

The results published in the paper are highly questionable since for the mentioned parameters the action space size is around 28M, and the results demonstrate
convergence within 1000 cycles. Nevertheless, after some corrections and reducing the spaces sizes, the algorithm works and converges.

##### The corrections are as follows:
1. UAV trajectory equation (6) was fixed from 𝒔(𝒕)=𝒔(𝑻𝒃)+𝒕/𝑻c(s(1)−𝒔(𝑻𝒃))to 𝒔(𝒕)=𝒔(𝑻𝒃)+(𝒕−𝑻𝒃)/(𝑻𝒄−𝑻𝒃)(𝒔(𝟏)−𝒔(𝑻𝒃)) 
2. The transmission state transition probabilities in equation (9) are changed as follows
    ![alt text](https://github.com/slangooo/decentralizedSenseAndSend/blob/master/probabilities_correction.png?raw=true)
3. LOS probability in equation (3) is not the same as in the 3GPP references (besides other discrepancies). 
   The exponential term in the case ri>rc is changed to exp((-ri/p0)(1-rc/ri))
   

##### Using the enhancements (i.e., limiting action space, and reward modeling) for both single-agent and opponent modeling algorithms yields the following results:
![alt text](https://github.com/slangooo/decentralizedSenseAndSend/blob/master/results/myplot.png?raw=true)
