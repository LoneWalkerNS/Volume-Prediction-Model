# Volume-Prediction Model

Model.py Explanation:-
I have created a machine learning model to use an ensemble of models to analyse historical volumes and assign weights to each of the models before using the same to predict the average traded volumes on a daily basis. The weigths are updated every day based on the acutal volumes traded.

Source Problem - https://github.com/RiskThinking/work-samples/blob/main/Data-Engineer.md 

GitIgnore

# Ignore files and directories generated by common development tools
/node_modules
/build
/dist
/.vscode
/.idea
/.DS_Store

# Ignore environment-specific files
.env
.env.local
.env.*.local
.env.*.production
.env.production.local

# Ignore log files and other temporary files
*.log
*.pid
*.swp
*.bak

# Ignore files generated during the build process
*.cache
*.css
*.html
*.js
*.map
*.png
*.svg
*.ico


Log files depends on the machine/server used. I would choose to get the files emailed at the end of every trading period to read and check on the spot.
