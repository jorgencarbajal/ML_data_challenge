Create a venv without a version control system...  
`uv init --vcs none`

or with vcs...  
`uv init`

If you later want to add version control, main is the branch  
`git init -b main`

create the `.gitignore` file and add necessary files

If you later want to push to github, create the repo then run this command. This runs a git command that nicknames the remote url as origin.    
`git remote add origin https://github.com/jorgencarbajal/ML_data_challenge.git`

This sends your local commits from your main branch to the GitHub repository named origin (-u is similar to --set-upstream which links your local main to your remote main).  
`git push -u origin main`