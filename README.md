# argkp-standalone

## Running
Download the state from [here](https://surfdrive.surf.nl/files/index.php/s/ymXl60rCMc2MPfQ) and extract into a `state/` folder. Install the `requirements.txt`. Then. run as follows:
```
python3 argkp.py <INFILE>
```

### File input format
As input, `argkp.py` requires a `.csv` file with the following columns:
1. `df['english']` -> an english sentence or comment.
2. `df['extracted_from']` -> 'pro' or 'con' depending on the stance of the comment.
3. `df['project_id']` -> a subclassification for project-dependent comments (also called "option"), can be selected using `--option <NUM>`.
4. `df['quality_scores']` -> an argument quality score between 0 and 1.