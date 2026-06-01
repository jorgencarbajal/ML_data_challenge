# Recommended Data Preparation Order

## Goal

Create a clean, defensible working dataset before selecting final baseline and recent comparison methods. The initial 68 features are candidate fields for exploration; they are not automatically the final modeling features.

---

## 1. Preserve the Original Dataset

* Keep the original 154-feature dataset unchanged.
* Create a separate working dataset containing the candidate fields selected for:

  * clustering;
  * association rule mining;
  * outcome construction;
  * auditing and validation.

This ensures that fields can be restored later if needed.

---

## 2. Check Identifiers and Duplicate Records

* Retain `reportkey` as a temporary audit field.
* Check whether `reportkey` values are unique.
* Investigate duplicated records before doing transformations or modeling.
* Keep `date` temporarily for timeline and missingness analysis.

---

## 3. Verify Outcome-Source Fields

Before creating outcome variables, confirm which injury and fatality fields are reliable and sufficiently populated.

Primary candidate fields:

* `totalkilledform55a`
* `totalinjuredform55a`

Temporary comparison fields:

* `totalkilledform57`
* `totalinjuredform57`
* `crossinguserskilled`
* `crossingusersinjured`
* `employeeskilled`
* `employeesinjured`
* `passengerskilled`
* `passengersinjured`

Questions to check:

* Are Form 55A totals populated consistently?
* Do Form 55A totals agree with Form 57 totals?
* Are the individual killed/injured fields redundant or useful for validation?

---

## 4. Create Outcome Variables

After verifying the source fields, create binary severity indicators:

* `fatality_present`

  * `1` if total killed is greater than 0;
  * `0` otherwise.

* `injury_present`

  * `1` if total injured is greater than 0;
  * `0` otherwise.

* `severe_outcome`

  * Defined from fatalities and/or injuries according to the final project decision.

Important:

* These outcome fields may be used as consequences in association rules.
* They should **not** be used as clustering inputs.
* For clustering, outcomes should only be examined after clusters are formed.

---

## 5. Check Missingness and Reporting Changes Over Time

For every candidate feature:

* calculate the missing-value count;
* calculate the missing-value percentage;
* inspect missingness by year.

Purpose:

* identify fields with too little usable information;
* determine whether fields were introduced or removed over time;
* decide whether to:

  * remove the feature;
  * restrict the analysis to certain years;
  * retain it only for a supporting analysis.

---

## 6. Inspect and Clean Categorical Features

For each categorical field:

* count unique values;
* calculate category frequencies and percentages;
* identify blank, unknown, or invalid values;
* identify categories that are extremely rare;
* compare coded and readable text versions of the same field.

Possible decisions:

* keep the original categories;
* combine rare or similar categories;
* keep only the readable text field;
* remove fields that are redundant or uninformative.

---

## 7. Transform Multi-Field and Time Variables

### Warning Devices

Inspect `crossingwarningexpanded1` through `crossingwarningexpanded12`.

These fields likely represent multiple devices present at the same crossing and should be transformed into binary indicators such as:

* `has_gate`
* `has_flashing_lights`
* `has_bell`
* `has_stop_sign`
* `has_crossbucks`

* 1 = `gates`
* 2 = `cantilever FLS`
* 3 = `standard FLS`
* 4 = `wig wags`
* 5 = `highway traffic signals`
* 6 = `audible`
* 7 = `cross bucks`
* 8 = `stop signs`
* 9 = `watchman`
* 10 = `flagged by crew`
* 11 = `other`
* 12 = `none`

This produces features that are easier to interpret and use in both clustering and association rules.

### Time Variables

Create derived time features from `date` and `time`, such as:

* `year`
* `season`
* `time_of_day`

Possible `time_of_day` categories:

* morning;
* afternoon;
* evening;
* night.

---

## 8. Inspect and Transform Numeric Features

Potential numeric fields include:

* `trainspeed`
* `estimatedvehiclespeed`
* `numbervehicleoccupants`
* `numberofcars`
* `temperature`
* `vehicledamagecost`

For each numeric field:

* check missingness;
* inspect extreme values and outliers;
* examine whether the values are strongly skewed;
* decide whether the feature is useful for each task.

Task-specific preparation:

* For clustering, numeric fields may need standardization.
* For association rules, numeric fields will likely need to be converted into categories or ranges, such as:

  * `train_speed_low`
  * `train_speed_medium`
  * `train_speed_high`

---

## 9. Check Task Feasibility and Finalize Feature Sets

### Association Rule Mining Feasibility

Check the frequency of:

* `fatality_present`
* `injury_present`
* `severe_outcome`

Purpose:

* determine whether severe outcomes occur frequently enough to support meaningful association rules;
* avoid selecting a recent association-rule method if the outcome is too rare for useful rules.

### Clustering Feasibility

Confirm that:

* severity variables are excluded from clustering inputs;
* transformed context variables are meaningful;
* the final feature representation matches the selected distance measure and clustering method.

### Final Feature Decision Table

Create a table assigning each candidate field one final status:

| Feature         | Keep for Clustering | Keep for Association Rules | Transform | Remove | Reason      |
| --------------- | ------------------: | -------------------------: | --------: | -----: | ----------- |
| Example feature |              Yes/No |                     Yes/No |    Yes/No | Yes/No | Explanation |

Only after this table is complete should the final baseline methods and recent comparison methods be selected.
