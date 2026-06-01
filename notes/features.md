# Project Questions & Initial grouping ideas

## Clustering (K-Medoids / PAM)

- Discover types/profiles of incidents without using severity to create the groups
- What distinct profiles of reported highway-rail incidents emerge from crossing conditions, highway-user characteristics, train characteristics, environmental conditions, and incident circumstances?
- The key idea is that clustering should probably use incident context features, not severity fields. Then, after cluster are formed, we can compare:
    - fatality and injury rates by cluster
    - average speeds by cluster
    - common warning-device combinations
    - whether some clusters correspond to pedestrians, trucks, poor visibility, or active-warning crossings.
- Initially we chose KMeans but the problem is the presence of continuous data. A resulting centroid by represent an artificial "average incident" that is harder to interpret.

### Feature Space

### Evaluation

## Association (Apriori association rule mining)

- Find context combinations associated with severe outcomes.
- Which combinations of incident context, crossing conditions, highway-user behavior, and train characteristics are frequently associated with sever reported outcomes?
- For this task, we may create an item such as:
    - `fatality_present`
    - `injury_present`
    - `severe_outcome`
- Then search for rules such as combinations involving:
    - highway-user type
    - warning devices
    - visibility or weather
    - train-speed category
    - highway-user action
    - equipment
- If association fails CART decision tree can be a backup.

### Feature Space

### Evaluation

## Which groups can server what purpose

1. Administrative / identifier fields

- Examples:
`reportkey`
`incidentnumber`
`otherincidentnumber`
`maintainenanceincidentnumber`
`url`
`gradecrossingid`
`railroad names and codes`

- These may help identify records or duplicates, but they are generally not useful as clustering or association-rule features because they are identifiers or high-cardinality labels.

2. Time and location

- Examples:
`date`
`month`
`hour`
`time`
`statename`
`countyname`
`cityname`
`highwayname`

- Possible use:
    - derive year, season, or time-of-day;
    - inspect whether incidents have changed over time;
    - interpret clusters geographically afterward.
- Caution: exact city, highway, or railroad names may dominate clusters without describing meaningful incident types.

3. Highway user and vehicle characteristics

- Examples:
`highwayuser`
`estimatedvehiclespeed`
`vehicledirection`
`highwayuserposition`
`userage`
`usergender`
`numbervehicleoccupants`

- These are potentially useful for defining different incident profiles, such as pedestrian incidents, truck incidents, or vehicle-occupant incidents.

4. Train and rail equipment characteristics

- Examples:
`equipmentinvolved`
`equipmentstruck`
`equipmenttype`
`tracktype`
`trackclass`
`numberoflocomotiveunits`
`numberofcars`
`trainspeed`

- These are strong candidates for clustering and association rules because they describe the rail-side context of the event.

5. Crossing safety devices and roadway conditions

- Examples:
`crossingwarningexpanded1` through `crossingwarningexpanded12`
`crossingwarning`
`crossingwarningexplanation`
`crossingwarninglocation`
`warningconnectedtosignal`
`crossingilluminated`
`roadwaycondition`
`whistleban`

- This is potentially one of the most interesting groups, but it will require preprocessing. The 12 warning fields appear to represent multiple warning devices present at a crossing, so they should probably be converted into binary features rather than treated as 12 separate ordered positions.

6. Environmental conditions

- Examples:
`temperature`
`visibility`
`weathercondition`
`viewobstruction`

- These could help identify incident contexts such as poor visibility, weather-related events, or obstructed-view situations.

7. Human actions and behavior

- Examples:
`highwayuseraction`
`driverpassedvehicle`
`userstruckbysecondtraincode`
`driverinvehicle`

- These may be especially useful for association rules because they represent event circumstances rather than administrative information.

8. Outcomes and severity

- Examples:
`crossinguserskilled`
`crossingusersinjured`
`employeeskilled`
`employeesinjured`
`passengerskilled`
`passengersinjured`
`totalkilledform57`
`totalinjuredform57`
`totalkilledform55a`
`totalinjuredform55a`
`vehicledamagecost`

- The dictionary says the Form 55A totals are the official fatality and injury numbers. These are likely the best candidates for defining severity, assuming the actual data confirms they are sufficiently populated.

# Feature Space

## Initial Feature Space pass

`reportkey`
`date`
`month`
`hour`
`time`
`highwayuser`
`estimatedvehiclespeed`
`highwayuserposition`
`userage`
`numbervehicleoccupants`
`equipmentinvolved`
`equipmentstruck`
`equipmenttype`
`tracktype`
`numberofcars`
`trainspeed`
`crossingwarningexpanded1`
`crossingwarningexpanded2`
`crossingwarningexpanded3`
`crossingwarningexpanded4`
`crossingwarningexpanded5`
`crossingwarningexpanded6`
`crossingwarningexpanded7`
`crossingwarningexpanded8`
`crossingwarningexpanded9`
`crossingwarningexpanded10`
`crossingwarningexpanded11`
`crossingwarningexpanded12`
`crossingwarning`
`crossingwarningexplanation`
`warningconnectedtosignal`
`crossingilluminated`
`roadwaycondition`
`temperature`
`visibility`
`weathercondition`
`viewobstruction`
`highwayuseraction`
`driverpassedvehicle`
`driverinvehicle`
`crossinguserskilled`
`crossingusersinjured`
`employeeskilled`
`employeesinjured`
`passengerskilled`
`passengersinjured`
`totalkilledform57`
`totalinjuredform57`
`totalkilledform55a`
`totalinjuredform55a`
`vehicledamagecost`

## Cluster Feature Space

highwayuser
highwayuserposition
`equipmentinvolved`
`equipmentstruck`
`equipmenttype`
tracktype
`trainspeed`
estimatedvehiclespeed
`crossingwarningexpanded1`
`crossingwarningexpanded2`
`crossingwarningexpanded3`
`crossingwarningexpanded4`
`crossingwarningexpanded5`
`crossingwarningexpanded6`
`crossingwarningexpanded7`
`crossingwarningexpanded8`
`crossingwarningexpanded9`
`crossingwarningexpanded10`
`crossingwarningexpanded11`
`crossingwarningexpanded12`
`crossingwarning`
`warningconnectedtosignal`
`crossingilluminated`
`roadwaycondition`
`visibility`
`weathercondition`
`viewobstruction`
highwayuseraction
driverpassedvehicle
driverinvehicle
hour/time
date/month = `season`

## Association Feature Space

- `fatality_present`, `injury_present`, or `severe_outcome` will be the consequence/output side of the rules, not regular input conditions.

`highwayuser`
`highwayuserposition`
`equipmentinvolved`
`equipmentstruck`
`equipmenttype`
tracktype
`trainspeed`
`estimatedvehiclespeed`
`crossingwarningexpanded1`
`crossingwarningexpanded2`
`crossingwarningexpanded3`
`crossingwarningexpanded4`
`crossingwarningexpanded5`
`crossingwarningexpanded6`
`crossingwarningexpanded7`
`crossingwarningexpanded8`
`crossingwarningexpanded9`
`crossingwarningexpanded10`
`crossingwarningexpanded11`
`crossingwarningexpanded12`
`crossingwarning`
`crossingwarningexplanation`
`warningconnectedtosignal`
`crossingilluminated`
`roadwaycondition`
`visibility`
`weathercondition`
`viewobstruction`
`highwayuseraction`
`driverpassedvehicle`
`driverinvehicle`
`hour/time`
date/month

`userage`
`numbervehicleoccupants`
`numberofcars`
`temperature`

## Outcome Feature Space

`totalkilledform55a`
`totalinjuredform55a`

Maybe temp...

`vehicledamagecost`

Temporary...

`totalkilledform57`
`totalinjuredform57`
`crossinguserskilled`
`crossingusersinjured`
`employeeskilled`
`employeesinjured`
`passengerskilled`
`passengersinjured`

## Audit Feature Space

`reportkey`
`date`

# Ideas in further reducing the union of the above spaces

After deciding on the feature space we need to explore the data in each feature to ensure there is actual value there.

Before doing so though, we will start by creating an EDA (exploratory data analysis). We need to take the original dataset and break it down into a smaller usable set. Consider this an initial first pass to get rid of all the crap and keep only what we truly believe may have value. This should look like...
$$\text{cluster features} \cup \text{association features} \cup \text{outcome fields} \cup \text{audit fields}$$
Then we can run the below tests to further define what has value.

1. missing-value percentage for all columns;
    - The goal here is to find out how much of each column actually has data. This can help us pinpoint things we can throw out and things that may be defined as part of rule #7.
2. unique-value counts for categorical columns;
    - Find out what types of fields have categorical data and how you are going to define those categories. You also need to consider how frequent certain categories are. See if there is imbalance or possibly columns with 99.9% of one category type. The goal is to inspect and decide.
3. date range and records per year;
    - This will simply provide a general dataset timeline, simple providing context
4. frequency of fatalities and injuries;
    - Creat usable binary outcome variables and checking whether they occur frequently enough
5. whether code/text pairs contain the same information;
    - remove redundant data
6. whether warning-device fields are populated consistently;
    - These represent multiple warning deviced present at a crossing, the goal is to convert them into indicators such as `has_gate`, `has_flashing_lights`...
7. which fields become missing because of form changes over time.
    - options: remove, restrict analysis to years where it exists, keep it only for a smaller supporting analysis

## Step by Step

1. Keep the original CSV unchanged.
2. Create an EDA working dataset containing:
    - the union of candidate clustering and association-rule features;
    - outcome fields: `totalkilledform55a`, `totalinjuredform55a`, and `vehicledamagecost`;
    - temporary audit fields: `reportkey`, `date`, and code/text column pairs.
3. Check for duplicate incidents using `reportkey`.
4. Create useful time variables from `date` and `time`:
    - `year`
    - `month` or `season`
    - `time_of_day`
5. Calculate the missing-value percentage for each field.
6. Check missingness by year to determine whether any fields were added, removed, or inconsistently collected over time.
7. Examine categorical features:
    - number of unique categories;
    - count and percentage of records in each category.
8. Compare code/text field pairs. If both contain the same information, keep only the readable text version.
9. Inspect the warning-device fields and convert `crossingwarningexpanded1` through `crossingwarningexpanded12` into meaningful binary device indicators.
10. Create outcome variables:
    - `fatality_present`
    - `injury_present`
    - possibly `severe_outcome`
11. Check outcome frequencies to determine whether severity-based association-rule mining is feasible.
12. Make final feature decisions:
    - keep for clustering;
    - keep for association rules;
    - transform;
    - remove.

# global variables
kept_features = [
    "Crossing Illuminated",
        18% Unknown, 3% missing
    "Driver In Vehicle",**********************
    "Driver Passed Vehicle",
        8% Unknown, 2% Missing
    "Equipment Involved",
        Several categories below 1%
    "Equipment Struck",**********************
    "Equipment Type",
        Several categories below 1%
    "Highway User",
        Several categories below 1%
    "Highway User Action",
        Several categories below 1%
    "Highway User Position",**********************
    "Track Type",**********************
    "View Obstruction",**********************
        Several categories below 1%
        Highly imbalanced
    "Visibility",**********************
    "Warning Connected To Signal",
        14% Unknown, 10% Missing
        substantial uncertainty and should be interpreted cautiously.
    "Weather Condition"**********************
]