import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


############################### DATA CLEANING FUNCTIONS ###############################

def clean_data(dataframe, speeding_column_name="speeding_status", lowercase_columns=True):
    
    dataframe = dataframe.copy(deep=True)

    # Drop rows where age is missing
    dataframe = dataframe.loc[~dataframe["AGE_IM"].isna(), :]
    
    # Reorganize the levels of the severity column
    dataframe = restructure_severity_column(dataframe=dataframe)
    
    # Bin the hours into Morning, Afternoon, Evening, Night
    dataframe["HOUR_BINNED"] = dataframe["HOUR_IM"].apply(lambda hour: create_binned_hours_feature(numeric_hour=hour))
    
    # Perform binning on the body type column
    dataframe["BODY_TYPNAME"] = dataframe["BODY_TYPNAME"].apply(lambda body_type: create_binned_vehicle_body_type_feature(body_type=body_type))
    
    # Add a feature for if someone was speeding (Yes, No, Unknown)
    dataframe[speeding_column_name] = dataframe.apply(lambda row: create_speeding_feature(row), axis='columns')

    # Remove state prefixes from region name
    dataframe["REGIONNAME"] = dataframe.loc[:,"REGIONNAME"].apply(lambda string: string.split()[0])

    # Trim the category names for our target class (Removes an unecessary suffix).
    dataframe['MAXSEV_IMNAME'] = dataframe['MAXSEV_IMNAME'].str[:-4]

    # Remove "Area" from urban indicator value
    dataframe["URBANICITYNAME"] = dataframe.loc[:,"URBANICITYNAME"].apply(lambda string: string.split()[0])

    # Lowercase column names
    if lowercase_columns:
        # Lowercase column names
        dataframe = dataframe.rename(columns=str.lower)
    
    return dataframe


def restructure_severity_column(dataframe):
    
    categories_to_drop = ['Died Prior to Crash*', 'Injured, Severity Unknown', 'No person involved']
    
    # Drop rows where MAXSEV has one of the categories above.
    dataframe = dataframe.loc[~dataframe["MAXSEV_IMNAME"].isin(categories_to_drop), :]
    
    # Convert all "Suspected Minor Injury (B)" to 'Possible Injury (C)'
    dataframe.loc[dataframe["MAXSEV_IMNAME"] == "Suspected Minor Injury (B)", "MAXSEV_IMNAME"] = 'Possible Injury (C)'
    
    return dataframe

def create_binned_hours_feature(numeric_hour, night_hours=[0,1,2,3,4,21,22,23], morning_hours=[5,6,7,8,9,10,11], 
                                afternoon_hours=[12,13,14,15], evening_hours=[16,17,18,19,20]):
    
    # NOTE: numeric_hour is a value from the HOUR_IM column
    
    # Default for night is 9pm-4:59am
    if numeric_hour in night_hours:
        return "Night"
    
    # Default for morning is 5am-11:59am
    elif numeric_hour in morning_hours:
        return "Morning"
    
    # Default for Afternoon is 12:00pm-3:59pm
    elif numeric_hour in afternoon_hours:
        return "Afternoon"
    
    # Default for evening is 4:00pm-8:59pm
    elif numeric_hour in evening_hours:
        return "Evening"

def create_binned_vehicle_body_type_feature(body_type):
    
    ## NOTE:
    ## Should do more binning here at some point
    ## but for now just creating this as a start, to mimic what Tavin did the other day
    
    binned_van_categories = ['Minivan (Chrysler Town and Country, Caravan, Grand Caravan, Voyager, Voyager, Honda-Odyssey, ...)',
                             'Large Van-Includes van-based buses (B150-B350, Sportsman, Royal Maxiwagon, Ram, Tradesman,...)',
                             'Unknown van type',
                             'Step van (GVWR greater than 10,000 lbs.)',
                             'Van-Based Bus GVWR greater than 10,000 lbs.',
                             'Other van type (Hi-Cube Van, Kary)',
                             'Step-van or walk-in van (GVWR less than or equal to 10,000 lbs.)']
    
    if body_type in binned_van_categories:
        return "Van"
    else:
        return body_type

def remove_observations_with_rare_column_value(df, column_name, min_observations_to_keep=100):
    
    # Count of the number of observations in each category
    category_counts = df[column_name].value_counts().sort_values()
    
    # Levels in column_name that occur less than min_observations_to_keep in the dataset
    sparse_category_levels = category_counts[category_counts < min_observations_to_keep].index.tolist()
    
    # Subset to remove rows where column_name has a value in sparse_category_levels
    df = df.loc[~df[column_name].isin(sparse_category_levels), :]
    
    return df

def create_speeding_feature(row):
    
    # Speed greater than 151 mph, you're speeding
    if row["TRAV_SP"] == 997:
        return 'speeding'
    
    # If we know they were going 95 mph or faster, calling that speeding
    # regardless of what the speed limit is, or if we even know ths speed limit.
    elif row["TRAV_SP"] < 152 and row["TRAV_SP"] >= 95:
        return 'speeding'
    
    # If the speed limit is unknown, speeding is unknown
    elif row["VSPD_LIM"] == 98 or row["VSPD_LIM"] == 99:
        return 'unknown'
    
    # If the traveling speed is unknown, speeding is unknown
    elif row['TRAV_SP'] == 998 or row['TRAV_SP'] == 999:
        return 'unknown'
    
    # If traveling faster than the speed limit, speeding
    elif row['TRAV_SP'] > row['VSPD_LIM']:
        return 'speeding'
    else:
        return 'not speeding'

    
def bin_body_type(df):
    
    df['body_type_binned']=df['BODY_TYPNAME']

    df['body_type_binned']=df['body_type_binned'].replace(to_replace=
                                        ['4-door sedan, hardtop',
                                       '2-door sedan,hardtop,coupe',
                                       '3-door coupe','Sedan/Hardtop, number of doors unknown'
                                        ],value=1)
    df['body_type_binned']=df['body_type_binned'].replace(to_replace=
                                        'Compact Utility (Utility Vehicle Categories \"Small" and \"Midsize\")',
                                         value=2)
    df['body_type_binned']=df['body_type_binned'].replace(to_replace=
                                                                                                                                                                                  ["Auto-based pickup (includes E1 Camino, Caballero, Ranchero, SSR, G8-ST, Subaru Brat, Rabbit Pickup)",
                                        "Light Pickup",
                                        "Unknown (pickup style) light conventional truck type",
                                        "Unknown light truck type",
                                        "Unknown light vehicle type (automobile,utility vehicle, van, or light truck)"
                                        ],value=3)
    df['body_type_binned']=df['body_type_binned'].replace(to_replace=
                                         ['Large utility (ANSI D16.1 Utility Vehicle Categories and "Full Size" and "Large")',
                                        'Utility Vehicle, Unknown body type'
                                         ],value=4)
    df['body_type_binned']=df['body_type_binned'].replace(to_replace=
                                        ['ATV/ATC [All-Terrain Cycle]',
                                        'Moped or motorized bicycle',
                                        'Motor Scooter',
                                        'Off-road Motorcycle',
                                        'Other motored cycle type (mini-bikes, pocket motorcycles "pocket bikes")',
                                        'Three-wheel Motorcycle (2 Rear Wheels)',
                                        'Two Wheel Motorcycle (excluding motor scooters)',
                                        'Unenclosed Three Wheel Motorcycle / Unenclosed Autocycle (1 Rear Wheel)',
                                        'Unknown motored cycle type',
                                        'Unknown Three Wheel Motorcycle Type'
                                        ],value=5)
    df['body_type_binned']=df['body_type_binned'].replace(to_replace=
                                        ['Station Wagon (excluding van and truck based)',
                                        'Utility station wagon (includes suburban limousines, Suburban, Travellall, Grand Wagoneer)'
                                        ],value=6)
    df['body_type_binned']=df['body_type_binned'].replace(to_replace=
                                        ['3-door/2-door hatchback',
                                        '5-door/4-door hatchback',
                                        'Hatchback, number of doors unknown'
                                        ],value=7)
    df['body_type_binned']=df['body_type_binned'].replace(to_replace=
                                        ['Cross Country/Intercity Bus',
                                        'Medium/Heavy Vehicle Based Motor Home',
                                        'Other Bus Type',
                                        'School Bus',
                                        'Transit Bus (City Bus)',
                                        'Unknown Bus Type'
                                        ],value=8)
    df['body_type_binned']=df['body_type_binned'].replace(to_replace=
                                        ['Cab Chassis Based (includes Rescue Vehicle, Light Stake, Dump, and Tow Truck)',
                                        'Medium/heavy Pickup (GVWR greater than 10,000 lbs.)',
                                        'Single-unit straight truck or Cab-Chassis (GVWR greater than 26,000 lbs.)',
                                        'Single-unit straight truck or Cab-Chassis (GVWR range 10,001 to 19,500 lbs.)',
                                        'Single-unit straight truck or Cab-Chassis (GVWR range 19,501 to 26,000 lbs.)',
                                        'Single-unit straight truck or Cab-Chassis (GVWR unknown)',
                                        'Truck-tractor (Cab only, or with any number of trailing unit; any weight)',
                                        'Unknown if single-unit or combination unit Heavy Truck (GVWR greater than 26,000 lbs.)',
                                        'Unknown if single-unit or combination unit Medium Truck (GVWR range 10,001 lbs. to 26,000 lbs.)',
                                        'Unknown medium/heavy truck type'
                                         ],value=9)
    df['body_type_binned']=df['body_type_binned'].replace(to_replace=
                                        'Convertible(excludes sun-roof,t-bar)',
                                           value=10)
    df['body_type_binned']=df['body_type_binned'].replace(to_replace=
                                        ['Large Van-Includes van-based buses (B150-B350, Sportsman, Royal Maxiwagon, Ram, Tradesman,...)',
                                        'Minivan (Chrysler Town and Country, Caravan, Grand Caravan, Voyager, Voyager, Honda-Odyssey, ...)',
                                        'Other van type (Hi-Cube Van, Kary)',
                                        'Step van (GVWR greater than 10,000 lbs.)',
                                        'Step-van or walk-in van (GVWR less than or equal to 10,000 lbs.)',
                                        'Unknown van type',
                                        'Van-Based Bus GVWR greater than 10,000 lbs.'
                                        ],value=11)
    df['body_type_binned']=df['body_type_binned'].replace(to_replace=
                                        ['Construction equipment other than trucks (includes graders)',
                                        'Farm equipment other than trucks',
                                        'Golf Cart',
                                        'Large Limousine-more than four side doors or stretched chassis',
                                        'Low Speed Vehicle (LSV) / Neighborhood Electric Vehicle (NEV)',
                                        'Not Reported',
                                        'Other or Unknown automobile type',
                                        'Other vehicle type (includes go-cart, fork-lift, city street sweeper dunes/swamp buggy)',
                                        'Recreational Off-Highway Vehicle',
                                        'Unknown body type',
                                        'Unknown truck type (light/medium/heavy)',
                                        ],value=12)
    return df

############################### END DATA CLEANING FUNCTIONS ###############################


############################### PLOTTING FUNCTIONS ###############################

def create_point_plot(dataframe, categorical_feature, continuous_feature, shading_feature=None, 
                      figsize=(18, 8), title=None, xlabel=None, ylabel=None, tick_rotation=0, tick_fontsize=16, 
                      xlab_fontsize=14, ylab_fontsize=14, title_fontsize=18, legend=True, legend_location="best", 
                      legend_frameon=True, palette=None, seaborn_style="whitegrid"):
    
    sns.set_style(seaborn_style)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, squeeze=True)
    
    
    sns.pointplot(x=categorical_feature, y=continuous_feature, hue=shading_feature, 
                  data=dataframe, ax=ax, palette=palette)
    
    if title is None:
        title = f"Distribution of {continuous_feature} for each level of {categorical_feature}"
    if xlabel is None:
        xlabel = f"Levels of {categorical_feature}"
    if ylabel is None:
        ylabel = f"Distribution of {continuous_feature}"
    
    # Annotate plot axes
    ax.set_title(f"{title}", fontsize=title_fontsize, weight='bold')
    ax.set_xlabel(xlabel, fontsize=xlab_fontsize, weight='bold')
    ax.set_ylabel(ylabel, fontsize=ylab_fontsize, weight='bold')
    ax.tick_params(axis='both', labelsize=tick_fontsize, labelrotation=tick_rotation)
    ax.legend(loc=legend_location, frameon=legend_frameon)
    
    if not legend:
        ax.legend_.remove()
    
    return ax

def create_strip_plot(dataframe, categorical_feature, continuous_feature, shading_feature=None, 
                      jitter=True, alpha=1, figsize=(18, 8), title=None, xlabel=None, ylabel=None,
                      tick_rotation=0, tick_fontsize=12, xlab_fontsize=12, ylab_fontsize=12, title_fontsize=16, 
                      legend=True, legend_location="best", legend_frameon=True, palette=None, seaborn_style="white"):
    
    sns.set_style(seaborn_style)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, squeeze=True)
    
    
    sns.stripplot(x=categorical_feature, y=continuous_feature, hue=shading_feature, 
                  data=dataframe, jitter=jitter, alpha=alpha, ax=ax,palette=palette)
    
    if title is None:
        title = f"Distribution of {continuous_feature} for each level of {categorical_feature}"
    if xlabel is None:
        xlabel = f"Levels of {categorical_feature}"
    if ylabel is None:
        ylabel = f"Distribution of {continuous_feature}"
    
    # Annotate plot axes
    ax.set_title(f"{title}", fontsize=title_fontsize, weight='bold')
    ax.set_xlabel(xlabel, fontsize=xlab_fontsize, weight='bold')
    ax.set_ylabel(ylabel, fontsize=ylab_fontsize, weight='bold')
    ax.tick_params(axis='both', labelsize=tick_fontsize, labelrotation=tick_rotation)
    ax.legend(loc=legend_location, frameon=legend_frameon)
    
    if not legend:
        ax.legend_.remove()
    
    return ax


# Basic count plot for factors
def plot_sorted_level_counts(dataframe, plotting_column, title, xlabel, ylabel, ylab_fontsize=12, title_fontsize=16,
                             xlab_fontsize=12, tick_fontsize=12, tick_rotation=45, ax=None, figsize=(10, 7), annot_fontsize=14, 
                             round_digits=4, add_annotations=True, annot_vshift=0.04, palette="husl", seaborn_style="white", 
                             title_weight='bold', xlab_weight='bold', ylab_weight='bold'):

    sns.set_style(seaborn_style)
    
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    # Order to plot the bars in
    plot_order = dataframe[plotting_column].value_counts().sort_values().index.tolist()

    # Add the plot
    sns.countplot(data=dataframe, x=plotting_column, ax=ax, order=plot_order, palette=palette)
        
    # Annotate plot axes
    ax.set_title(f"{title}", fontsize=title_fontsize, weight=title_weight)
    ax.set_xlabel(xlabel, fontsize=xlab_fontsize, weight=xlab_weight)
    ax.set_ylabel(ylabel, fontsize=ylab_fontsize, weight=ylab_weight)
    ax.tick_params(axis='both', labelsize=tick_fontsize, labelrotation=tick_rotation)
    
    if add_annotations:
        _add_count_plot_annotations(ax=ax, 
                                    num_observations=dataframe.shape[0], 
                                    annot_vshift=annot_vshift, 
                                    round_digits=round_digits, 
                                    annot_fontsize=annot_fontsize)
    
    return 

### Plotting Functions
# Plot the distribution of categorical features, and the percentage of observations in each level.
def plot_feature_counts_by_grouping_level(dataframe, plotting_column, grouping_column, num_rows=1, num_cols=2,
                                          title_fontsize=16, xlab_fontsize=12, ylab_fontsize=12, tick_fontsize=12, tick_rotation=45,
                                          annot_fontsize=14, round_digits=4, add_annotations=True, figsize=(18, 7), annot_vshift=0.04,
                                          palette="husl", seaborn_style="white"):
    sns.set_style(seaborn_style)
    
    # List of the levels in feature column
    group_feature_levels = dataframe[grouping_column].unique().tolist()

    # List of the levels in the plotting column, used to handle one-off circumstances where text_to_color_map
    # Is missing colors for some factor levels because the plotting_column did not have any observations
    # of that type in the first group of grouping_column
    all_plot_column_levels = dataframe[plotting_column].unique().tolist()
    
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize, squeeze=False)
    
    # For each plot we are going to add
    for index, level in enumerate(group_feature_levels):
        
        plot_df = dataframe.loc[dataframe[grouping_column]==level, :]
        
        title = f"Distribution of {plotting_column}\nWhen {grouping_column}={level}\nTotal Number of Observations={plot_df.shape[0]}"
        xlabel = plotting_column
        ylabel = f"Count of {plotting_column}"
        
        # Grab the grid location for this plot
        col = index % num_cols
        row = index // num_cols
        
        plot_sorted_level_counts(dataframe=plot_df, 
                                 plotting_column=plotting_column, 
                                 ax=axs[row][col],
                                 title=title, 
                                 xlabel=xlabel, 
                                 ylabel=ylabel, 
                                 ylab_fontsize=ylab_fontsize, 
                                 title_fontsize=title_fontsize, 
                                 xlab_fontsize=xlab_fontsize,
                                 tick_fontsize=tick_fontsize, 
                                 tick_rotation=tick_rotation, 
                                 annot_fontsize=annot_fontsize, 
                                 round_digits=round_digits, 
                                 add_annotations=add_annotations, 
                                 annot_vshift=annot_vshift)
        
        # Text to color map lets us ensure that, across all plots, the bars with
        # the same x-label will also be the same color.
        if row == col == 0:
            text_to_color_map = _get_text_to_color_map(ax=axs[row][col], palette=palette, all_plot_column_levels=all_plot_column_levels)
        
        # Ensure the bars with the same tick label have the same color, across all plots.
        _set_patch_colors(ax=axs[row][col], text_to_color_map=text_to_color_map)
    

    return plt.tight_layout()

# Helper function to plot_feature_counts_by_grouping_level
def _add_count_plot_annotations(ax, num_observations, annot_vshift, round_digits, annot_fontsize):

    # Annotate the percentages on top of the bars
    for p in ax.patches: 
            
        # Percentage is the ratio of the bar height over the total people
        percentage = f"{round((100 * (p.get_height() / num_observations)), round_digits)}%"

        # Annotate on the left edge of the bar
        x = p.get_x()
        
        # Annotate just above the top of the bar
        y = p.get_y() + p.get_height() + annot_vshift
            
        #Perform annotation
        ax.annotate(percentage, (x,y), fontsize=annot_fontsize, fontweight='bold')
        
    return

# Helper function to plot_feature_counts_by_grouping_level
def _get_text_to_color_map(ax, all_plot_column_levels, palette="husl"):
    
    # X-axis text objections
    x_text_objects = ax.get_xticklabels()
    
    # Color palette with one color for each level in the feature being plotted
    color_palette = sns.color_palette(palette=palette, n_colors=len(all_plot_column_levels))
    
    # Map axis label text --> color
    text_to_color_map = {x_text_obj.get_text():color for x_text_obj, color in zip(x_text_objects, color_palette[:len(x_text_objects)])}

    # Handling the one-off situation where the color map is going to be too short
    if len(text_to_color_map) < len(all_plot_column_levels):
        missing_levels = [column_name for column_name in all_plot_column_levels if column_name not in text_to_color_map]
        for index, level in enumerate(missing_levels):
            text_to_color_map[level] = color_palette[index]
    
    return text_to_color_map

# Helper function to plot_feature_counts_by_grouping_level
def _set_patch_colors(ax, text_to_color_map):
    
    x_text_objects = ax.get_xticklabels()
    x_tick_positions = ax.get_xticks()
    
    text_to_coordinate_map = {x_text_obj.get_text():x_pos for x_text_obj, x_pos in zip(x_text_objects, x_tick_positions)}
    
    # Iterate across the x-tick text labels
    for text, text_x_pos in text_to_coordinate_map.items():
        
        # Grab the color associated with this text label
        color = text_to_color_map[text]
        
        # Iterate over the patches object, find the one at the location
        # that needs to be color next, then color it.
        for p in ax.patches:
            patch_x_pos = p.get_x()
            patch_width = p.get_width()
            patch_text_location = patch_x_pos + (patch_width/2)
            
            # If this is the location that needs to be colored next
            if patch_text_location == text_x_pos:
                p.set_color(c=color)
    
    return 



############################### END PLOTTING FUNCTIONS ###############################



############################### UTILITY FUNCTIONS ###############################

import os

def find_all_float_dtypes(directory="./data/"):
    
    float_columns = {}
    
    files = os.listdir(directory)
    
    for file_name in files:
        
        file_path = os.path.join(directory, file_name)
        
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='iso-8859-1', low_memory=False)
        
        float_column_names = df.select_dtypes(include=["float_", "float16", "float32", "float64"]).columns.tolist()
        
        float_columns[file_name] = float_column_names
        
    return float_columns

    
############################### END UTILITY FUNCTIONS ###############################