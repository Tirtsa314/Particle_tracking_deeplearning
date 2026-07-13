""" 
author: Tirtsa den Haan 
06-07-2026

HTML clicker
"""


#%%
import os
import webbrowser
import pandas as pd




#change this to the location of the HTML_corrections folder:
os.chdir(r"C:\Particle_Tracking_Deeplearning_Tirtsa\structured_code\HTML_corrections")



from html_functions import make_particle_clicker_html, reading_nd2 ,apply_particle_clicker_edits

#%%


csv_path = r"c:\Analysis_images\Hydrazine 010\final_particles_voronoi.csv" #path of detections csv
image_path =  r"c:\Users\Public\Hydrazine 010.nd2" #path of nd2 image file

frame,arr8 = reading_nd2(image_path, frame_index=-1)

initial_df = pd.read_csv(csv_path)

html_path = make_particle_clicker_html(
    image=arr8,
    particles=initial_df,         
    output_html=r"C:\Analysis_images\particle_clicker.html", #can change to an analysis folder to save the html
    output_csv_name="particle_manual_edits.csv",
    x_col="x",
    y_col="y",
    id_col="particle_id",
    point_radius=4,
)


webbrowser.open(html_path)

#%%

#downloaded file from the html:
edits_csv=r"c:\Users\DenHaan\Downloads\particle_manual_edits.csv"


#new df that can be saved to:
save_path=r"C:\Analysis_images\particles_corrected_test_06-07.csv"
final_df = apply_particle_clicker_edits(
    particles=initial_df,
    edits_csv=edits_csv,
    x_col="x",
    y_col="y",
    id_col="particle_id",
    save_path=save_path,
)

