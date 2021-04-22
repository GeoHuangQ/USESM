# USESM
Urban Spatially Epidemic Simulation Model
# Overall model
We adopted the definition of the compartment in the SEIR model, classifying the population into four agents—susceptible, exposed, infection, and recovered in the multi-agent system. The study area was divided into several sub-regions, and mobile phone signaling data and point-of-interest (POI) data were used to calculate the decision probability of every agent moving, staying, and being infected in the sub-regions. The roulette algorithm translated these decision probabilities into specific agent actions; the decision with higher probability has a greater chance of being executed, while the decision with a smaller probability still has a slight chance. We set the sub-regions of the model as Beijing streets which comprise 128 units; data from the outbreak was used to validate the model’s accuracy and usefulness.
# Model framework
This model integrates the initialization module as well as the infection movement module and the epidemic transmission module. In the initialization module, the study area is divided into several sub-regions and generates four types of agents (S: susceptible, E: exposed, I: infection, R: recovered) in the sub-regions based on the infections, incubation periods, and population at the initial time (Time = 1). The initial module generates a SIER dataset to store all data of the model including current time, infection period, incubation period, agent status, quantity, etc. In the infection movement module, the movement possibility of each infection is calculated, and a binary roulette is built based on this probability to determine whether the infection can move. For each infection requiring movement, the model calculates the probability of moving to each sub-region and uses a multivariate roulette to determine the destination. Moreover, the model calculates the probability that the infection will stay at the destination; if this decision probability is successfully converted into agent action, the infection will stay at the destination until removed; otherwise, the infection will return to the origin sub-region on the same day. In the epidemic transmission module, POI and mobile phone signaling data are used to correct the basic reproduction number (R0) in different sub-regions, then calculates the number of the next generation exposed through a random draw from a Poisson distribution (mathematical expectation is corrected R0) for each infection. The temporarily moved infections will return to the origin sub-region after this step to avoid interference with subsequent epidemic transmission simulation. At the end of the epidemic transmission module, Bernoulli trials (mathematical expectations are the reciprocal of incubation period and infection period) are used for each exposed person and infections to acquire the next generation infections and removed persons. Finally, the model checks whether the SEIR data reaches the termination condition: if not, the SEIR dataset will proceed to the next iteration (Time = time +1); if so, the model will output the SEIR dataset for the entire simulation period.
# Input options and formats

- pop_path: Path of parameters data (Each type of parameter is a sheet in EXCEL)
>   pat_locator: a table with 4 variables (pat_name;pat_region;pat_id;area).
>>  "pat_name": the names of different sub-regions
>>  
>>  "pat_region": the names of different region
>>  
>>  "pat_id":  the IDs of different sub-regions
>>  
>>  "area": the area of different sub-regions
>>  
>   initial_inf: the number of initial infections in each sub-region
>   
>   initial_exp: the number of initial suspected infections in each sub-region
>   
>   control_df_out: mobility restriction rate of population leaving each sub-region
>   
>   control_df_in：mobility restriction rate of active population in each sub-area
>   
>   R0: daily basic reproduction number
>   
>   rec_rate: daily recovery rate
>   
>   day_pd: date
>   
>   den_poi: density of different POI types in each sub-area

- pop_data: Path of daily population data (CSV)
>    To count the daily population of every sub-region, we defined the base station where a mobile phone user appeared most frequently between 10 p.m. and 6 a.m. as the user’s home location, then counted the number of home locations in each sub-region as the daily population.
>
>    daily population data with 3 variables (pat_id,date,pop)
>
>>  "pat_id": the IDs of different sub-regions
>>
>>  "date": date
>>
>>  "pop": the number of population in each sub-area


- flow_data: Path of Human mobility data (CSV)
>    Human mobility refers to the number of people moving from region i to region n on each day. To quantify human mobility, we performed trajectory segmentation (time threshold over 30 min, distance threshold over 200 m) on each trajectory in the mobile phone signaling data, and extracted the midpoint of each segmentation as the stay location. Moreover, we used these stay locations as the starting and ending point of each movement of the mobile phone user, then counted the number of points in each sub-region to reflect the number of population movements within and between sub-regions.
>
>    Human mobility data with 6 variables (fr_pat,to_pat,date,move_prop,distance,area)
>
>>  "fr_pat": the origin sub-region ID of human mobility flow
>>
>>  "to_pat": the destination sub-region ID of human mobility flow
>>
>>  "date": date
>>
>>  "move_prop": the number of population movements within and between sub-regions
>>
>>  "distance": the distance of population movements within and between sub-regions

# Data sharing

The epidemiological data were obtained from the government website of Beijing. We purchased the Mobile phone signaling data (May 5 to June 30, 2020) from the service provider (China Mobile). Our data purchase agreement with China Mobile prohibits us from sharing these data with third parties, but interested parties can contact China Mobile to make the same data purchase. All data has been approved by the ethics board.

**Contact**: HuangQiang; huangq@lreis.ac.cn