"""
The goal with this form is to sift through the data and build a better understanding for what is there and what is not there.
"""

import pandas as pd


# global variables
kept_features = [
    "Report Key",
    "Date",
    "Month",
    "Hour",
    "Time",
    "Highway User",
    "Estimated Vehicle Speed",
    "Highway User Position",
    "User Age",
    "Number Vehicle Occupants",
    "Equipment Involved",
    "Equipment Struck",
    "Equipment Type",
    "Track Type",
    "Number of Cars",
    "Train Speed",
    "Crossing Warning Expanded 1",
    "Crossing Warning Expanded 2",
    "Crossing Warning Expanded 3",
    "Crossing Warning Expanded 4",
    "Crossing Warning Expanded 5",
    "Crossing Warning Expanded 6",
    "Crossing Warning Expanded 7",
    "Crossing Warning Expanded 8",
    "Crossing Warning Expanded 9",
    "Crossing Warning Expanded 10",
    "Crossing Warning Expanded 11",
    "Crossing Warning Expanded 12",
    "Signaled Crossing Warning",
    "Crossing Warning Explanation",
    "Warning Connected To Signal",
    "Crossing Illuminated",
    "Roadway Condition",
    "Temperature",
    "Visibility",
    "Weather Condition",
    "View Obstruction",
    "Highway User Action",
    "Driver Passed Vehicle",
    "Driver In Vehicle",
    "Crossing Users Killed",
    "Crossing Users Injured",
    "Employees Killed",
    "Employees Injured",
    "Passengers Killed",
    "Passengers Injured",
    "Total Killed Form 57",
    "Total Injured Form 57",
    "Total Killed Form 55A",
    "Total Injured Form 55A",
    "Vehicle Damage Cost"
]


def main_reduction(df) -> pd:
    """
    This function reduces the 154 features into 51. Essentially getting rid of all the crap that I define to be useless.
    """

    # Confirm that every chosen feature exists in the original dataset
    missing_features = [feature for feature in kept_features if feature not in df.columns]

    if missing_features:
        print("Missing features error")
    else:
        # get the copy of the reduces feature space to save into a csv
        reduced_df = df[kept_features].copy()
    
    return reduced_df


def remove_duplicates(df):
    cleaned_df = df.drop_duplicates(subset="Report Key", keep="first").copy()
    return cleaned_df


def print_duplicate_report_keys(df):
    duplicate_rows = df[df["Report Key"].duplicated(keep=False)]

    if duplicate_rows.empty:
        print("No duplicated Report Key values found.")
    else:
        duplicate_rows.to_csv("data/duplicates.csv", index=False)
        print(f"Saved {duplicate_rows.shape[0]} duplicated rows to data/duplicates.csv")


def audit_outcome_totals(df):
    # the fields we are going to inspect
    outcome_fields = [
        "Total Killed Form 55A",
        "Total Injured Form 55A",
        "Total Killed Form 57",
        "Total Injured Form 57"
    ]

    # a list of dictionaries for each of the fields we will be inspecting.
    coverage_rows = []

    for field in outcome_fields:
        # convert the column of the df into a Series and fill the missing values with NaN
        values = pd.to_numeric(df[field], errors="coerce")

        # add the dicitionary entry with all the field information
        coverage_rows.append({
            "Field": field,
            "Total Rows": len(df),
            "Missing Count": values.isna().sum(),
            "Missing Percent": values.isna().mean() * 100,
            "Zero Count": (values == 0).sum(),
            "Greater Than Zero Count": (values > 0).sum(),
            "Greater Than Zero Percent": (values > 0).mean() * 100
        })

    # convert that list of dictionaries into a dataframe
    coverage_df = pd.DataFrame(coverage_rows)

    # after obtaining the summaries, we need to compare 55A and 57
    comparison_pairs = [
        ("Total Killed Form 55A", "Total Killed Form 57"),
        ("Total Injured Form 55A", "Total Injured Form 57")
    ]

    agreement_rows = []

    for form55a_field, form57_field in comparison_pairs:
        # separate the columns for comparison
        form55a = pd.to_numeric(df[form55a_field], errors="coerce")
        form57 = pd.to_numeric(df[form57_field], errors="coerce")

        # series that will hold true/false values for comparison
        comparable = form55a.notna() & form57.notna()
        agree = comparable & (form55a == form57)
        disagree = comparable & (form55a != form57)

        # add in the summaries
        agreement_rows.append({
            "Form 55A Field": form55a_field,
            "Form 57 Field": form57_field,
            "Rows Compared": comparable.sum(),
            "Agreement Count": agree.sum(),
            "Disagreement Count": disagree.sum(),
            "Agreement Percent": agree.sum() / comparable.sum() * 100 if comparable.sum() > 0 else 0,
            "Missing Either Field": (~comparable).sum()
        })

    # convert to df
    agreement_df = pd.DataFrame(agreement_rows)

    # store them in csv
    coverage_df.to_csv("data/outcome_coverage_audit.csv", index=False)
    agreement_df.to_csv("data/outcome_agreement_audit.csv", index=False)


def add_outcome_variables(df):
    df["fatality_present"] = (df["Total Killed Form 55A"] > 0).astype(int)
    df["injury_present"] = (df["Total Injured Form 55A"] > 0).astype(int)
    
    return df


def audit_missingness(df):
    # go through the entire dataset and replace missing values with missing value marker
    audit_df = df.replace(r"^\s*$", pd.NA, regex=True).copy()

    total_rows = len(audit_df)

    overall_missingness_df = pd.DataFrame({
        "Feature": audit_df.columns,
        "Total Rows": total_rows,
        "Missing Count": audit_df.isna().sum().values,
        "Missing Percent": (audit_df.isna().mean() * 100).round(2).values,
        "Non-Missing Count": audit_df.notna().sum().values
    })

    # sort by missing percentage
    overall_missingness_df = overall_missingness_df.sort_values(
        "Missing Percent",
        ascending=False
    )

    # create an additional column for year, convert and extract the year portion of the entries
    audit_df["Year"] = pd.to_datetime(
        audit_df["Date"],
        errors="coerce"
    ).dt.year
 
    yearly_rows = []

    # group rows by year
    for year, year_df in audit_df.groupby("Year"):
        for feature in df.columns:
            yearly_rows.append({
                "Year": int(year),
                "Feature": feature,
                "Total Rows In Year": len(year_df),
                "Missing Count": year_df[feature].isna().sum(),
                "Missing Percent": round(year_df[feature].isna().mean() * 100, 2)
            })

    missingness_by_year_df = pd.DataFrame(yearly_rows)

    # Save audit outputs
    overall_missingness_df.to_csv(
        "data/overall_missingness_audit.csv",
        index=False
    )

    missingness_by_year_df.to_csv(
        "data/missingness_by_year_audit.csv",
        index=False
    )

    print("Saved: data/overall_missingness_audit.csv")
    print("Saved: data/missingness_by_year_audit.csv")


def main():
    # load the data
    df = pd.read_csv("data/original_data.csv", low_memory=False)

    # functions that reduces the data 154 to 51 and remove duplicates
    df = main_reduction(df=df)
    df = remove_duplicates(df=df)
    # audit_outcome_totals(df)
    df = add_outcome_variables(df=df)
    # audit_missingness(df=df)

    

    df.to_csv("data/pass_1.csv", index=False)


if __name__ == "__main__":
    main()