# USESM
Urban Spatially Epidemic Simulation Model
# Overall model
We adopted the definition of the compartment in the SEIR model, classifying the population into four agents—susceptible, exposed, infection, and recovered in the multi-agent system. The study area was divided into several sub-regions, and mobile phone signaling data and point-of-interest (POI) data were used to calculate the decision probability of every agent moving, staying, and being infected in the sub-regions. The roulette algorithm translated these decision probabilities into specific agent actions; the decision with higher probability has a greater chance of being executed, while the decision with a smaller probability still has a slight chance. We set the sub-regions of the model as Beijing streets which comprise 128 units; data from the outbreak was used to validate the model’s accuracy and usefulness.
# Model framework
This model integrates the initialization module as well as the infection movement module and the epidemic transmission module. In the initialization module, the study area is divided into several sub-regions and generates four types of agents (S: susceptible, E: exposed, I: infection, R: recovered) in the sub-regions based on the infections, incubation periods, and population at the initial time (Time = 1). The initial module generates a SIER dataset to store all data of the model including current time, infection period, incubation period, agent status, quantity, etc. In the infection movement module, the movement possibility of each infection is calculated, and a binary roulette is built based on this probability to determine whether the infection can move. For each infection requiring movement, the model calculates the probability of moving to each sub-region and uses a multivariate roulette to determine the destination. Moreover, the model calculates the probability that the infection will stay at the destination; if this decision probability is successfully converted into agent action, the infection will stay at the destination until removed; otherwise, the infection will return to the origin sub-region on the same day. In the epidemic transmission module, POI and mobile phone signaling data are used to correct the basic reproduction number (R0) in different sub-regions, then calculates the number of the next generation exposed through a random draw from a Poisson distribution (mathematical expectation is corrected R0) for each infection. The temporarily moved infections will return to the origin sub-region after this step to avoid interference with subsequent epidemic transmission simulation. At the end of the epidemic transmission module, Bernoulli trials (mathematical expectations are the reciprocal of incubation period and infection period) are used for each exposed person and infections to acquire the next generation infections and removed persons. Finally, the model checks whether the SEIR data reaches the termination condition: if not, the SEIR dataset will proceed to the next iteration (Time = time +1); if so, the model will output the SEIR dataset for the entire simulation period.
# Input options and formats

pop_path:Path of parameter data (Each type of variable is a sheet in EXCEL)
>   pat_locator: a table with 4 variables (pat_name;pat_region;pat_id;area).
>>  "pat_name": the names of different streets in Beijing.
>>  
>>  "pat_region": the names of different districts in Beijing.
>>  
>>  "pat_id":  the IDs of different streets in Beijing.
>>  
>>  "area": the area of different streets in Beijing.
>>  
>   initial_inf: The number of initial infections in each sub-region
>   
>   initial_exp: The number of initial suspected infections in each sub-region
>   
>   control_df_out: mobility restriction rate of population leaving each sub-region
>   
>   control_df_in：mobility restriction rate of active population in each sub-area
>   
>   R0: Daily basic reproduction number
>   
>   rec_rate: Daily recovery rate
>   
>   day_pd: date
>   
>   den_poi: density of different POI types in each sub-area




> sheet_name
- pop_data:
- flow_data:

# Data sharing
The epidemiological data were obtained from the government website of Beijing. We purchased the Mobile phone signaling data (May 5 to June 30, 2020) from the service provider (China Mobile). Our data purchase agreement with China Mobile prohibits us from sharing these data with third parties, but interested parties can contact China Mobile to make the same data purchase. All data has been approved by the ethics board.

**Contact**: HuangQiang; huangq@lreis.ac.cn