## Critical Thinking / Crutch > 1 ?

This md file will be for mapping my thought process and the assistance I am seeking of chat to see how much critical thinking I am doing vs chat.

### Setup

I started out by setting up the project folder, the only help in this was the syntax for adding a vcs, `git remote add origin http...`. Other than that setup and download was primarily on my own.

Another part of the setup involved setting up the Latex template for the report. It would be very time consuming to sit and attempt to understand what every file in the template folder from the provided website was doing. Instead I had chat summarize what each was for and what was necessary to keep and discard.

### Structure

After setting the environment we now had to plan out the structure of the project. This involved setting up a roadmap that will correctly hit all the targets I need. I initially missed the idea of setting up the questions that would lead to the goal of the project. What is the purpose. I am new to the world of machine learning so didn't really have that thinking cap on. I will continue to attempt to develop that critical thinking skill. 

Aside from the questions I had some reasoning backwards in thinking I should pick my task before sifting threw the features and building an understanding of the features that I was working with. It obviously now makes more sense to have an understanding of the data I am working with before deciding what types of tasks work best.

### Task selection / feature reduction

I did have chat heavily help me in looking through the features and pinpointing which would work best for the clustering and association ideas I had in mind. Again... deciding before understanding the data.

I next decided it was best to look through the filtered 68 features and pick a bucket that works for clustering and another that works for association. I thought to pick and move one but was again dumbfounded in missing the idea of an *exploratory check*. The idea here is that we only want to include features that actually bring value. If the feature is entirely empty with no data, it makes sense to remove it use.

Chat helped me pick the features that best work for each of the two tasks we chose. I initially missed a few and added others. The best approach to have done this completely alone was to go through all 154, refine, then again further refine into cluster/association buckets... ain't nobody got time for that.

I submitted my notes and so far the features.md file to have an ultimate best idea for what tasks I should focus on. After initially thinking that KMeans was going to be it, chat recommended KMedoids. Apparently KMeans works better for continuos data and since this data is not continuous chat recommended medoids. I wouldnt have noticed that but now I know. I will have to ensure I further explore and explain why Kmedoids is better.

Ok I initially thought that I neede to first find the more recent methods after settling on KMedoids and Apriori. But I was again deviated in my thought process that it was better to do that initial pass in the EDA process to see what the true structure of the data is.

Pass1 data refining: reduce to 51 columns, remove duplicates, ensure that the deaths in the data set will be meaningful enough to produce reasonable results. After running the audit function inside the pass1 file, we have determined that the values from 55A and 57 agree over 98% of the time, no need for additional changes. Also 8.3% of the data has a more than 0 fatalities and 23.4% of the data has injuries, this shows that clustering and association is possible, right?