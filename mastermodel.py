# %% [markdown]
# # 3D Packing Capacity Problem

# %% [markdown]
# #### Importing packages

# %%
from sys import argv 
import os
import numpy as np 
import pandas as pd
from amplpy import AMPL
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import matplotlib.patches as mpatches
from PIL import Image
import math

# %% [markdown]
# ### Notations
# 
# - $I=\{1,2,...,N\}$ is the set of SKUs, where N denotes the total number of SKUs to be packed.
#   
# - $t$  is the number of copies of the DFC
# 
# - $T$ is the set of copies of the DFC
# 
# - The $i^{th}$ SKU is characterized by its length $l_i$, width $w_i$, height $h_i$, weight $m_i$. $\forall i \in I$
# 
# - The DFC is characterized by its length $L$, width $W$, height $H$ and maximum weight carrying capacity $G$
# 
# - $lr^+_{ii′} = 1$, if $i^{th}$ SKU is placed on left-hand side of $i′^{th}$ SKU; otherwise, $lr^+_{ii′} = 0$.
# - $bf^+_{ii′} = 1$, if $i^{th}$ SKU is placed behind the $i′^{th}$ SKU;    otherwise, $bf^+_{ii′} = 0$.
# - $bt^+_{ii′} = 1$, if $i^{th}$ SKU is placed on below the $i′^{th}$ SKU; otherwise, $bt^+_{ii′} = 0$.
# - $lr^-_{ii′} = 1$, if $i^{th}$ SKU is placed on right-hand side of $i′^{th}$  SKU; otherwise, $lr^-_{ii′} = 0$.
# - $bf^-_{ii′} = 1$, if $i^{th}$ SKU is placed in front of $i′^{th}$ SKU; otherwise, $bf^-_{ii′} = 0$.
# - $bt^-_{ii′} = 1$, if $i^{th}$ SKU is placed above the $i′^{th}$ SKU; otherwise, $bt^-_{ii′} = 0$.
# 
# - $u_k$ is a binary variable, which denotes whether $k_{th}$ copy of the DFC.
# 
# 
# - $s_{ik}$ is a binary variable to denote, whether the $i_{th}$ SKU is packed in $k_{th}$ copy of
# the DFC.
# 
# - The variables $x_i$, $y_i$, $z_i$ denotes the coodinated of the left-bottom-back corner of
# $i_{th}$ SKU.
# 
# - The Variables $l^x_i$ , $l^y_i$ , $l^z_i$ denotes whether the length edge of ith SKU is parallel to X-axis, Y-axis or Z-axis.
# 
# - Similarly, the variables $w^x_i$ , $w^y_i$ , $w^z_i$ denotes whether the width edge of ith SKU is parallel to X-axis, Y-axis or Z-axis, and the variables $h^x_i$ , $h^y_i$ , $h^z_i$ denotes whether the height edge of $i_{th}$ SKU is parallel to X-axis, Y-axis or Z-axis.
# 
# - $B$ is the bound on dfc dimensions.
# 
# 
# ### Formulation
# 
# The objective is to minimize the total vacant spaces or unused volume within
# the assigned DFC or, equivalently, the total volume of the chosen DFC for
# packing. The latter is easier to model. The overall model is as follows: 
# 
# $$
# \begin{align}
# \min \quad 
# \begin{cases}
# L & \text{if cubic DFC is enforced} \\
# \sum_{k \in T} u_k LWH - \sum_{i \in I} l_i w_i h_i & \text{otherwise}
# \end{cases} \\
# 
# \text{s.t.} \quad 
# & \sum_{k\in T} s_{ik} = 1 
# && \forall i \in I \\
# 
# & u_k \ge s_{ik}
# && \forall i \in I,\; \forall k \in T_j \\
# 
# & l_i^x + l_i^y + l_i^z = 1
# && \forall i \in I \\
# 
# & w_i^x + w_i^y + w_i^z = 1
# && \forall i \in I \\
# 
# & h_i^x + h_i^y + h_i^z = 1
# && \forall i \in I \\
# 
# & l_i^x + w_i^x + h_i^x = 1
# && \forall i \in I \\
# 
# & l_i^y + w_i^y + h_i^y = 1
# && \forall i \in I \\
# 
# & l_i^z + w_i^z + h_i^z = 1
# && \forall i \in I \\
# 
# & x_i + l_i^x l_i + w_i^x w_i + h_i^x h_i
# \le L + (1-s_{ik})M
# && \forall i\in I,\forall k\in T \\
# 
# & y_i + l_i^y l_i + w_i^y w_i + h_i^y h_i
# \le W + (1-s_{ik})M
# && \forall i\in I,\forall k\in T \\
# 
# & z_i + l_i^z l_i + w_i^z w_i + h_i^z h_i
# \le H + (1-s_{ik})M
# && \forall i\in I,\forall k\in T \\
# 
# & \sum_{i\in I} s_{ik} m_i \le G
# && \forall k\in T \\
# 
# & x_i + l_i^x l_i + w_i^x w_i + h_i^x h_i
# \le x_{i'} + (1-lr_{ii'}^{+})M
# && i<i',\forall i,i'\in I \\
# 
# & x_{i'} + l_{i'}^x l_{i'} + w_{i'}^x w_{i'} + h_{i'}^x h_{i'}
# \le x_i + (1-lr_{ii'}^{-})M
# && i<i',\forall i,i'\in I \\
# 
# & y_i + l_i^y l_i + w_i^y w_i + h_i^y h_i
# \le y_{i'} + (1-bf_{ii'}^{+})M
# && i<i',\forall i,i'\in I \\
# 
# & y_{i'} + l_{i'}^y l_{i'} + w_{i'}^y w_{i'} + h_{i'}^y h_{i'}
# \le y_i + (1-bf_{ii'}^{-})M
# && i<i',\forall i,i'\in I \\
# 
# & z_i + l_i^z l_i + w_i^z w_i + h_i^z h_i
# \le z_{i'} + (1-bt_{ii'}^{+})M
# && i<i',\forall i,i'\in I \\
# 
# & z_{i'} + l_{i'}^z l_{i'} + w_{i'}^z w_{i'} + h_{i'}^z h_{i'}
# \le z_i + (1-bt_{ii'}^{-})M
# && i<i',\forall i,i'\in I \\
# 
# & lr_{ii'}^{+} + lr_{ii'}^{-}
# + bf_{ii'}^{+} + bf_{ii'}^{-}
# + bt_{ii'}^{+} + bt_{ii'}^{-}
# \ge s_{ik} + s_{i'jk} - 1
# && i<i',\forall i,i'\in I,\forall k\in T\\
# 
# & u_p \ge u_q
# \qquad
# && \forall
# p,q \in T: q = p + 1\\
# 
# & L = W & \text{deactivate if cubic DFC is not enforced}\\
# & W = H & \text{deactivate if cubic DFC is not enforced}\\
# 
# & L \le B,\; W \le B,\; H \le B & \text{deactivate if DFC is unbounded}\\
# 
# & u_k \le \sum_{i \in I} s_{ik}
# \qquad \forall k \in T & \text{deactivate if cubic DFC is not enforced}
# 
# \end{align}
# $$

# %%
## Visualization function 

# %%
def plot_cube(lower, upper, ax, col, opacity): # function to add a cube having one coordinate at lower and its opposite coordinate at upper
    '''
    “Fix 2 coordinates → vary 1 coordinate → get a line”

    Example:

    Fix y, z → vary x → edge along x
    Fix x, z → vary y → edge along y
    Fix x, y → vary z → edge along z
    '''
    s = []      # list of corner points of cuboid
    for i in range(len(upper)): # iterate over axes → x, y, z
        l = [i for i in range(3)]
        l.remove(i)                     # get the other two axes
        for j in range(4):         # loop over 4 combinations (binary 00, 01, 10, 11) due to bitwise operator '&'
            cl = upper.copy()       # one endpoint
            cl[l[0]]= upper[l[0]] if (j&1)>0 else lower[l[0]]        
            cl[l[1]]= upper[l[1]] if (j&2)>0 else lower[l[1]]     # choose corners based on binary mask
            cl2 = cl.copy()
            cl2[i] = lower[i]          # create second point → edge along axis i
            if list(cl2) not in s:
                s.append(list(cl2))
            if list(cl) not in s:
                s.append(list(cl))
            ax.plot3D(*zip(cl2, cl), color = col)     # draw edge between two points
    faces = []
    for i in range(3): # iterate over axes → x, y, z
        l = []      # l and u each has one face for every loop
        u = []
        for j in s:
            if (upper[i]==j[i]):
                u.append(j)
            if lower[i]==j[i]:
                l.append(j)
        l.sort()
        l[0], l[1] = l[1], l[0]
        u.sort()
        u[0], u[1] = u[1], u[0]
        faces.append(l)
        faces.append(u)
    fc = art3d.Poly3DCollection(faces, alpha = opacity)
    fc.set_color(col)
    ax.add_collection(fc)
    return ax

def visualise(SKU_coordinate, sku, dfc, id, title, argv): # function to plot configuration of all SKUs into a single DFC 
    # It outputs a gif for that packed dfc and images of each dfc placed in that dfc individually
    # this function takes sku cordinates in that dfc, sku dimensions, dfc dimensions , title , and command line arguments for naming output files
    # Expects terminal input: python script.py [data_filename] [dfc_type] [bounded/unbounded] [later_use]
    # mypath creates a unique folder inside /results/ based on command line arguments
    mypath = f"./results/{argv[1]}_{argv[2]}_{argv[3]}_{argv[4]}"   # path where to save the image
    #print(mypath)
    #exit(0)
    if not os.path.isdir(mypath): # makes a path if mypath doesn't exist
        os.makedirs(mypath)
    fig = plt.figure("1")
    ax = fig.add_subplot(projection='3d')
    # ax.set_aspect("equal")

    # Generate a discrete rainbow colormap based on the number of items (SKUs)
    # Each item gets a unique 'slice' of the HSV color wheel
    cmap = plt.cm.get_cmap('hsv', len(sku)+1)

    ax = plot_cube([0,0,0], dfc, ax, 'black', 0.05)
    Images = []
    # Iterates through each SKU to draw them step-by-step
    for x in range(len(sku)):
        ax.set_title(title[x])
        ax = plot_cube(SKU_coordinate[x], np.array(SKU_coordinate[x])+np.array(sku[x]), ax, cmap(x+1), 0.2)
        # fig = plt.figure("1")
        # Define filename: e.g., /results/ID_0.png, /results/id_1.png
        fname = f"/{str(id)}_{str(x)}.png"
        #print(fname, mypath, mypath+fname)
        # Save the current state of the 3D plot as a high-res image
        fig.savefig(mypath+fname)
        print("\npng saved to file: %s"%mypath+fname)
        #Images.append(Image.open(str(id)+"_"+str(x)+".png"))
        # Open the saved image via PIL and store in memory for GIF creation
        Images.append(Image.open(mypath+fname))
    # plt.show()
    # fig = plt.figure("1")
    # fig.show()
    # print(sku)
    plt.close() # Close plot window to free up system memory
    gifname = f"/{str(id)}.gif"
    print(gifname, mypath, mypath+gifname)
    print(Images)
    # Compile the list of PNGs into a single animated GIF
    # duration=1000 means 1 second per frame; loop=0 means infinite looping
    Images[0].save(mypath+gifname, save_all = True, append_images = Images[1:], duration = 1000, loop = 0)
    print("\ngif saved to file: %s"%mypath+gifname)
    #exit(0)

# %% [markdown]
# #### Make model

# %%
# getting parameters from command line arguments
csv = f"{argv[1]}.csv"     # argument 1 is the data file of sku's

# dfc is required to be cubic or cuboid?
cubic_dfc=0
if argv[2] == "cubic":
    cubic_dfc=1
else:
    pass

# dfc dimension is required to be bounded or not?
bound_dfc=0
if argv[3] == "bounded":
    bound_dfc=1
else:
    pass

# Data formatting(Pipeline)
try:
    sku_df = pd.read_csv(csv, encoding='utf-8')
except UnicodeDecodeError:
    sku_df = pd.read_csv(csv, encoding='cp1252')

# Select only needed columns
sku_df = sku_df[['Length','Width','Height','Actual Weight', 'Qty']]

# Expand rows by Qty column
sku_df = sku_df.loc[sku_df.index.repeat(sku_df['Qty'])].reset_index(drop=True)

# Drop Qty column
sku_df = sku_df.drop(columns=['Qty'])

# Add SKU column with incrementing values
sku_df['sku'] = range(1, len(sku_df) + 1)

# Rename 'Actual Weight' to 'Weight'
sku_df = sku_df.rename(columns={'Actual Weight': 'Weight'})

print(sku_df.to_string())


# Initialize ampl object
ampl = AMPL()

ampl.eval(
    """
    set SKU;                             
    set Copy;                            # set of copies of the DFC with elements equal to the number of SKU
    set axis= {'x', 'y', 'z'};
    set dim= {'Length', 'Width', 'Height'};

    param dim_sku {SKU, dim};
    param sku_Weight {SKU};
    param bound{dim};
    param M;    

    var dim_dfc {dim} >=0;
    var dfc_Weight >=0;

    var relative_position_left {i in SKU, l in SKU : i<l} binary;
    var relative_position_right {i in SKU, l in SKU : i<l} binary;
    var relative_position_back {i in SKU, l in SKU : i<l} binary;
    var relative_position_front {i in SKU, l in SKU : i<l} binary;
    var relative_position_below {i in SKU, l in SKU : i<l} binary;
    var relative_position_above {i in SKU, l in SKU : i<l} binary;

    var copy_used {k in Copy} binary;
    var sku_in_copy {i in SKU, k in Copy} binary;
    var sku_position {SKU, axis} >=0;

    var Length_orientation {SKU, axis} binary;
    var Width_orientation {SKU, axis} binary;
    var Height_orientation {SKU, axis} binary;

    minimize vacant_space:
        sum{k in Copy} copy_used[k]*dim_dfc['Length']*dim_dfc['Width']*dim_dfc['Height'] - sum{i in SKU} dim_sku[i,'Length']*dim_sku[i,'Width']*dim_sku[i,'Height']; 


    s.t. c1 {i in SKU}: 
    sum{k in Copy} sku_in_copy[i,k] = 1;

    s.t. c2 {i in SKU}:
    Length_orientation[i,'x'] + Length_orientation[i,'y'] + Length_orientation[i,'z'] = 1;

    s.t. c3 {i in SKU}:
    Width_orientation[i,'x'] + Width_orientation[i,'y'] + Width_orientation[i,'z'] = 1;

    s.t. c4 {i in SKU}:
    Height_orientation[i,'x'] + Height_orientation[i,'y'] + Height_orientation[i,'z'] = 1;

    s.t. c5 {i in SKU}:
    Length_orientation[i,'x'] + Width_orientation[i,'x'] + Height_orientation[i,'x'] = 1;

    s.t. c6 {i in SKU}:
    Length_orientation[i,'y'] + Width_orientation[i,'y'] + Height_orientation[i,'y'] = 1;

    s.t. c7 {i in SKU}:
    Length_orientation[i,'z'] + Width_orientation[i,'z'] + Height_orientation[i,'z'] = 1;

    s.t. c8 {i in SKU, k in Copy}:  
    sku_position[i,'x'] + Length_orientation[i,'x']*dim_sku[i,'Length'] + Width_orientation[i,'x']*dim_sku[i,'Width'] + Height_orientation[i,'x']*dim_sku[i,'Height'] <= dim_dfc[ 'Length'] + (1- sku_in_copy[i,k])*M;
    
    s.t. c9 {i in SKU, k in Copy}: 
    sku_position[i,'y'] + Length_orientation[i,'y']*dim_sku[i,'Length'] + Width_orientation[i,'y']*dim_sku[i,'Width'] + Height_orientation[i,'y']*dim_sku[i,'Height'] <= dim_dfc[ 'Width'] + (1- sku_in_copy[i,k])*M;
    
    s.t. c10 {i in SKU, k in Copy}: 
    sku_position[i,'z'] + Length_orientation[i,'z']*dim_sku[i,'Length'] + Width_orientation[i,'z']*dim_sku[i,'Width'] + Height_orientation[i,'z']*dim_sku[i,'Height'] <= dim_dfc[ 'Height'] + (1- sku_in_copy[i,k])*M;

    s.t. c11 {i in SKU, k in Copy}: 
    copy_used[k] >= sku_in_copy[i,k];

    s.t. c12 {k in Copy}:
    sum{i in SKU} sku_in_copy[i,k]*sku_Weight[i] <= dfc_Weight;

    s.t. c13 {(i,l) in {i in SKU, l in SKU : i<l}}:
    sku_position[i,'x'] + Length_orientation[i,'x']*dim_sku[i,'Length'] + Width_orientation[i,'x']*dim_sku[i,'Width'] + Height_orientation[i,'x']*dim_sku[i,'Height'] <= sku_position[l,'x'] + (1- relative_position_left[i,l])*M;
    
    s.t. c14 {(i,l) in {i in SKU, l in SKU : i<l}}:
    sku_position[l,'x'] + Length_orientation[l,'x']*dim_sku[l,'Length'] + Width_orientation[l,'x']*dim_sku[l,'Width'] + Height_orientation[l,'x']*dim_sku[l,'Height'] <= sku_position[i,'x'] + (1- relative_position_right[i,l])*M;

    s.t. c15 {(i,l) in {i in SKU, l in SKU : i<l}}:
    sku_position[i,'y'] + Length_orientation[i,'y']*dim_sku[i,'Length'] + Width_orientation[i,'y']*dim_sku[i,'Width'] + Height_orientation[i,'y']*dim_sku[i,'Height'] <= sku_position[l,'y'] + (1- relative_position_back[i,l])*M;
    
    s.t. c16 {(i,l) in {i in SKU, l in SKU : i<l}}:
    sku_position[l,'y'] + Length_orientation[l,'y']*dim_sku[l,'Length'] + Width_orientation[l,'y']*dim_sku[l,'Width'] + Height_orientation[l,'y']*dim_sku[l,'Height'] <= sku_position[i,'y'] + (1- relative_position_front[i,l])*M;

    s.t. c17 {(i,l) in {i in SKU, l in SKU : i<l}}:
    sku_position[i,'z'] + Length_orientation[i,'z']*dim_sku[i,'Length'] + Width_orientation[i,'z']*dim_sku[i,'Width'] + Height_orientation[i,'z']*dim_sku[i,'Height'] <= sku_position[l,'z'] + (1- relative_position_below[i,l])*M;
    
    s.t. c18 {(i,l) in {i in SKU, l in SKU : i<l}}:
    sku_position[l,'z'] + Length_orientation[l,'z']*dim_sku[l,'Length'] + Width_orientation[l,'z']*dim_sku[l,'Width'] + Height_orientation[l,'z']*dim_sku[l,'Height'] <= sku_position[i,'z'] + (1- relative_position_above[i,l])*M;
    
    s.t. c19 {i in SKU, l in SKU, k in Copy : i<l}:
    relative_position_left[i,l] + relative_position_right[i,l] + relative_position_back[i,l] + relative_position_front[i,l] + relative_position_below[i,l] + relative_position_above[i,l] >= sku_in_copy[i,k] + sku_in_copy[l,k] -1;

    s.t. c20 {p in Copy, q in Copy : q = p+1}:
    copy_used[p] >= copy_used[q];

    s.t. cube_condition1:
        dim_dfc['Length'] = dim_dfc['Width'];

    s.t. cube_condition2:
        dim_dfc['Width'] = dim_dfc['Height'];

    s.t. dfc_bound {m in dim}:
        dim_dfc[m] <= bound[m];
    """
    )

# Feeding data to the model
ampl.set['SKU'] = sku_df['sku']
ampl.set['Copy'] = sku_df['sku']    # maximum number of copies of DFC can be equal to the number of SKU at most

ampl.param['dim_sku'] = sku_df.set_index('sku').loc[:,['Length', 'Width', 'Height']]
ampl.param['sku_Weight'] = sku_df.set_index('sku')['Weight']
ampl.param['M'] = float(sku_df[['Length', 'Width', 'Height']].max(axis=1).sum())

# Enforcing dfc type(Cubic or Cuboidal)
if cubic_dfc==1:
    pass  # keep constraints
else:
    ampl.get_constraint('cube_condition1').drop()
    ampl.get_constraint('cube_condition2').drop()


# Enforcing bound on dfc dimension
if bound_dfc==1:   # keep constraints
    ampl.param['bound'] = [100,100,100] # max from dfc length, width , height 
else:
    ampl.get_constraint('dfc_bound').drop()

# Solve 
ampl.option['solver'] = 'gurobi'
ampl.option['gurobi_options'] = 'timelimit=50 mipgap=0.05 outlev=1'
ampl.solve()

solve_result = ampl.get_value('solve_result')
print(f"Solve result: {solve_result}")
print(f"Objective:    {ampl.get_value('vacant_space')}")


# ============================================================
# 🔍 IIS ANALYSIS BLOCK (Irreducible Infeasible Subsystem)
# ============================================================
# This block triggers only if the solver fails to find a solution. 
# It identifies the minimal set of conflicting constraints.
if "infeasible" in str(solve_result).lower():
    solver = ampl.get_option("solver").lower()
    
    # Enable IIS (Irreducible Infeasible Subsystem) finding based on active solver
    if "gurobi" in solver:
        ampl.option["gurobi_options"] = "iisfind=1"
    elif "cplex" in solver:
        ampl.option["cplex_options"] = "iis=1"
    else:
        print("⚠️ IIS computation not supported for this solver.")
        solver = None

    if solver:
        # Re-solve is required to trigger the IIS detection routines
        ampl.solve() 
        from amplpy import OutputHandler

        # Capture AMPL's internal printf output into a list for Python processing
        class CaptureOutput(OutputHandler):
            def __init__(self): self.lines = []
            def output(self, kind, msg): self.lines.append(msg)

        capture = CaptureOutput()
        try:
            ampl.set_output_handler(capture)
            # Print IIS members in a pipe-delimited format for easy parsing
            ampl.eval("""
                for {j in 1.._ncons} if _con[j].iis != 'non' then 
                    printf "CON | %-6s | %s\\n", _con[j].iis, _conname[j];
                for {j in 1.._nvars} if _var[j].iis != 'non' then 
                    printf "VAR | %-6s | %s\\n", _var[j].iis, _varname[j];
            """)
        finally:
            # Restore default output handler to prevent memory leaks
            ampl.set_output_handler(None)

        # Convert raw string records into a structured pandas DataFrame
        records = []
        for line in capture.lines:
            if line.startswith(("CON |", "VAR |")):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3:
                    records.append({"kind": parts[0], "iis": parts[1], "name": parts[2]})

        if records:
            df = pd.DataFrame(records)
            # Group constraints (e.g., 'c13[1,5]' and 'c13[2,4]' both become 'c13')
            # This allows us to see which logic rule is failing globally.
            df["group"] = df["name"].str.replace(r"\[.*\]", "", regex=True)

            # Identify which constraint groups appear most frequently in the conflict
            conflict_summary = (
                df[df["kind"] == "CON"]
                .groupby("group").size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )

            print("\n=== Top Conflicting Constraint Groups ===")
            print(conflict_summary.to_string(index=False))

            # Heuristic Auto-Diagnosis based on your model's specific notations
            def diagnose(group):
                rules = {
                    "c13": "Overlap: Left-side positioning is physically impossible.",
                    "c14": "Overlap: Right-side positioning is physically impossible.",
                    "c15": "Overlap: Behind/Front positioning is clashing.",
                    "c17": "Overlap: Below/Above positioning is clashing.",
                    "c8": "Boundary: SKU exceeds Container Length.",
                    "c9": "Boundary: SKU exceeds Container Width.",
                    "c10": "Boundary: SKU exceeds Container Height.",
                    "c12": "Weight: Total SKU weight exceeds DFC capacity.",
                    "orientation": "Inconsistent orientation rules (L/W/H assignments)."
                }
                for key, msg in rules.items():
                    if key in group.lower(): return msg
                return "General constraint conflict detected."

            print("\n=== Likely Issues (Auto-Diagnosis) ===")
            for g in conflict_summary.head(3)["group"]:
                print(f"[{g}] -> {diagnose(g)}")

            # 📊 Clashing Frequency Plot
            # This visualizes which parts of the model are 'fighting' each other most
            plt.figure(figsize=(10, 5))
            conflict_summary.set_index("group")["count"].plot(kind="bar", color='salmon')
            plt.title("IIS Conflict Distribution (Clashing Frequency)")
            plt.xlabel("Constraint Group")
            plt.ylabel("Frequency in IIS")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

        else:
            print("IIS identified by solver, but no members were captured.")

else:
    # --- Results block: only runs when solved ---
    dim_dfc_df = ampl.get_variable('dim_dfc').to_pandas()
    dfc_weight_df = ampl.get_variable('dfc_Weight').to_pandas()

    copy_used_df     = ampl.var['copy_used'].get_values().to_pandas()
    sku_in_copy_df   = ampl.var['sku_in_copy'].get_values().to_pandas()
    sku_position_df  = ampl.var['sku_position'].get_values().to_pandas()
    Length_orientation_df = ampl.var['Length_orientation'].get_values().to_pandas()
    Width_orientation_df = ampl.var['Width_orientation'].get_values().to_pandas()
    Height_orientation_df = ampl.var['Height_orientation'].get_values().to_pandas()

    def fix_2d_index(df, col2_name):
        """
        Convert AMPL multi-index DataFrame into a clean tabular format.

        AMPL variables like sku_position, orientation, sku_in_copy are returned
        as pandas DataFrames with a multi-index structure:
            (index0, index1) → value

        Example:
            index0 = SKU
            index1 = axis / copy

        This function:
        1. Resets the index → converts index levels into columns
        2. Renames columns to meaningful names:
            - first index → 'sku'
            - second index → 'axis' or 'copy'
        3. Returns a clean DataFrame for further processing (pivot, filtering, plotting)
        """

        # Step 1: Convert multi-index to columns
        df = df.reset_index()

        # Step 2: Rename generic column names to meaningful ones
        df = df.rename(columns={
            df.columns[0]: 'sku',        # SKU index
            df.columns[1]: col2_name     # 'axis' or 'copy'
        })

        return df


    # --- Apply transformation to all AMPL outputs ---

    # SKU positions: (sku, axis → x/y/z)
    sku_position_df = fix_2d_index(sku_position_df, 'axis')

    # Orientation variables: (sku, axis)
    Length_orientation_df = fix_2d_index(Length_orientation_df, 'axis')
    Width_orientation_df  = fix_2d_index(Width_orientation_df, 'axis')
    Height_orientation_df = fix_2d_index(Height_orientation_df, 'axis')

    # SKU assignment: (sku, copy)
    sku_in_copy_df = fix_2d_index(sku_in_copy_df, 'copy')

    # Scalar counts and volumes
    copies_used    = int((copy_used_df['copy_used.val'] > 0.5).sum())

    # dim_dfc_df has one row with L, W, H columns — product across columns
    vals = dim_dfc_df['dim_dfc.val']
    dfc_volume_each = float(vals.prod())
    dfc_volume_total = copies_used * dfc_volume_each

    # Enforcing dfc type(Cubic or Cuboidal)
    if cubic_dfc==1:
        # total sku volume
        sku_volume_total = (sku_df['Length'] * sku_df['Width'] * sku_df['Height']).sum()

        # packing efficiency = total_sku / total_dfc
        packing_efficiency = sku_volume_total / dfc_volume_total
    else:
        # vacant_space = total DFC volume used − total SKU volume (your objective)
        # so packing efficiency = 1 - vacant / total_dfc
        packing_efficiency = 1 - ampl.get_value('vacant_space') / dfc_volume_total

    # Gurobi reports its own solve time via this suffix
    gurobi_time = ampl.get_value('_total_solve_time')
    print(f"Solve time (Gurobi): {gurobi_time:.4f} seconds")

    print(f"Packing efficiency: {packing_efficiency * 100:.2f}%")

    print("\n=== DFC dimension ===")        
    print(dim_dfc_df)

    print("\n=== DFC weight capacity ===")
    print(dfc_weight_df)

    print("\n=== Copies used ===")
    print(copy_used_df[copy_used_df['copy_used.val'] > 0.5])

    print("\n=== SKU assignments ===")
    print(sku_in_copy_df[sku_in_copy_df['sku_in_copy.val'] > 0.5])

    print("\n=== SKU positions ===")
    print(sku_position_df)

    print("\n=== Length orientation ===")
    print(Length_orientation_df[Length_orientation_df['Length_orientation.val'] > 0.5])

    print("\n=== Width orientation ===")
    print(Width_orientation_df[Width_orientation_df['Width_orientation.val'] > 0.5])

    print("\n=== Height orientation ===")
    print(Height_orientation_df[Height_orientation_df['Height_orientation.val'] > 0.5])

    # sku_in_copy has (sku, copy) instead of (sku, axis)
    sku_in_copy_df = sku_in_copy_df.reset_index()
    sku_in_copy_df.columns = [c.lower() for c in sku_in_copy_df.columns]

    # 1. Prepare SKU Coordinates (Starting points)
    # We pivot the dataframe so we have one row per SKU with columns x, y, z
    pos = sku_position_df.pivot(index='sku', columns='axis', values='sku_position.val')

    # --- START OF VISUALIZATION INTEGRATION ---
    
    # 1. Pivot the orientation dataframes so we can easily access them by SKU
    L_orient = Length_orientation_df.pivot(index='sku', columns='axis', values='Length_orientation.val')
    W_orient = Width_orientation_df.pivot(index='sku', columns='axis', values='Width_orientation.val')
    H_orient = Height_orientation_df.pivot(index='sku', columns='axis', values='Height_orientation.val')

    # 2. Identify which container copies were actually used
    active_copies = copy_used_df[copy_used_df['copy_used.val'] > 0.5].index.tolist()

    for k in active_copies:
        # 3. Filter SKUs assigned to THIS specific container 'k'
        assigned_skus_df = sku_in_copy_df[(sku_in_copy_df['copy'] == k) & 
                                            (sku_in_copy_df['sku_in_copy.val'] > 0.5)]
        sku_ids = assigned_skus_df['sku'].tolist()

        if not sku_ids:
            continue

        # 4. Filter coordinates
        filtered_coords = pos.loc[sku_ids, ['x', 'y', 'z']].values.tolist()

        # 5. CALCULATE ROTATED DIMENSIONS (The missing link)
        filtered_dims = []
        for s_id in sku_ids:
            # Get original dimensions
            orig = sku_df[sku_df['sku'] == s_id].iloc[0]
            L, W, H = orig['Length'], orig['Width'], orig['Height']
            
            # Calculate the effective dimension along each axis based on orientation variables
            # x_dim = (Is Length on X? * L) + (Is Width on X? * W) + (Is Height on X? * H)
            dim_x = (L_orient.loc[s_id, 'x'] * L) + (W_orient.loc[s_id, 'x'] * W) + (H_orient.loc[s_id, 'x'] * H)
            dim_y = (L_orient.loc[s_id, 'y'] * L) + (W_orient.loc[s_id, 'y'] * W) + (H_orient.loc[s_id, 'y'] * H)
            dim_z = (L_orient.loc[s_id, 'z'] * L) + (W_orient.loc[s_id, 'z'] * W) + (H_orient.loc[s_id, 'z'] * H)
            
            filtered_dims.append([dim_x, dim_y, dim_z])

        # 6. Prepare container metadata
        container_dims = [
            dim_dfc_df.loc['Length', 'dim_dfc.val'],
            dim_dfc_df.loc['Width', 'dim_dfc.val'],
            dim_dfc_df.loc['Height', 'dim_dfc.val']
        ]
        
        gif_filename_id = f"Container_Copy_{k}"
        step_titles = [f"Box {k} | SKU {s_id}" for s_id in sku_ids]

        print(f"Rendering Container {k} with {len(sku_ids)} SKUs (Rotations applied)...")
        visualise(filtered_coords, filtered_dims, container_dims, gif_filename_id, step_titles, argv)
        
    # --- END OF VISUALIZATION INTEGRATION ---
    # visualise(currentSKU_Coordinates, currentSKU_Dimensions, currentDFC, str(argv[1])+"_"+dfc_df.iloc[j-1]['Name'], title)
    # visualise(currentSKU_Coordinates, currentSKU_Dimensions, currentDFC, f"{argv[1]}_{dfc_instance_name}", title, argv)

# %% [markdown]
# ## Call the model function

# %%
#model('VmeasureData.csv')