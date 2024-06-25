#!/bin/bash
#SBATCH --nodes=1                   #####Nb de noeuds
#SBATCH --ntasks-per-node=6        ##### Nb de coeurs/noeuds ici 24
#SBATCH --time=24:00:00             ##### temps demandé maximum de mon calcul
#SBATCH --mail-user=Giovanni.CATALANI2@isae-supaero.fr #### mon mail
#SBATCH --mail-type=FAIL,END                   #### mail quand c'est fini ou erreur
#SBATCH --job-name=INR_Airfrans                   #### mon calcul
#SBATCH -o    slurm.%j.out                     #### sorti standard STDOUT
#SBATCH -e    slurm.%j.err                     #### sorti erreur STDERR
#SBATCH --partition=gpu                        #### la partition
#SBATCH --gres=gpu:1

####  loading python you want 2.7 or 3.7 ###
module purge
module load python/3.7
module list

##### Activate your environement ######
source deactivate
source activate aifrans_challenge


##### launch your program #####
python load_data.py
