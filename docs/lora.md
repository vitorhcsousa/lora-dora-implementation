
`rank`: hyperparameter that controls the inner dimension of the matrices $A$ and $B$. This control the number of 
additional  parameters introduced by LoRa and it's crucial to determine the balance between model adaptability  and 
parameter efficiency.

`alpha`: is a scaling hyperparameter applied to the output of the low-rank adaptation. It essentially controls the 
extent to which the adapted layer's output is allowed to influence the original output opf the layer being adapted. 
This can be seen as a awy to regulate the impact of the low-rank-adaptation on the layer's output. 


