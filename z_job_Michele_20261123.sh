# ********************************************************* My Ubuntu Office Laptop ********************************************************* # 
source /home/pooja4593/Downloads/Softwares/miniconda3/etc/profile.d/conda.sh	# Load conda (based on your actual Miniconda install path)
#conda activate /home/pooja4593/Downloads/Softwares/miniconda3/envs/condacurraun			# This works too
# ******************************************************************************************************************************************* #


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& IIT Goa: Physics &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& #
#source /home/Pooja/z_Softwares/miniconda3/etc/profile.d/conda.sh		# Load conda (based on your actual Miniconda install path)
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& #


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% IIT Goa: SKD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
#source /home/pooja/z_Softwares/miniconda3/etc/profile.d/conda.sh		# Load conda (based on your actual Miniconda install path)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


# ================================================================ JYU: Kone ================================================================ #
#source /nashome5/poojapo/miniconda3/etc/profile.d/conda.sh			# Load conda (based on your actual Miniconda install path)
# Activate your conda environment (adjust this if you've named your environment)
#conda activate /nashome5/poojapo/miniconda3/envs/condacurraun					# Full path to unnamed env
# =========================================================================================================================================== #


# +++++++++++++++++++++++++++++++++++++++++++++++++ JYU: GPUs (Thingfish, Tarandot, Calaf) ++++++++++++++++++++++++++++++++++++++++++++++++++ #
#source /n/work00/poojapo/z_Softwares/miniconda3/etc/profile.d/conda.sh		# Load conda (based on your actual Miniconda install path)	
#conda activate /n/work00/poojapo/z_Softwares/miniconda3/envs/condacurraun		# May fail
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #



conda activate condacurraun



# Launch in background; nohup automatically creates nohup.out
#nohup python -u MASTER_CODE_2_LY1.py &
nohup python MASTER_CODE.py &


# Capture process ID
PID=$! 
echo $PID > z_jobfileID.pid 

echo "================================================="
echo "Michele code script running in background with PID: $PID"
echo "Log file: nohup.out"
echo "Track output live with: tail -f nohup.out"
echo "================================================="
