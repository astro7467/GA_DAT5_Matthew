#source ~/.bashrc

source activate

if [ -f '$HOME/anaconda3/bin/jupyter' ]; then
    "$HOME/anaconda3/bin/jupyter" notebook --notebook-dir="$HOME/GitHubDAT5"
    exit 0
fi

if [ -f '$HOME/anaconda2/bin/jupyter' ]; then
    "$HOME/anaconda2/bin/jupyter" notebook --notebook-dir="$HOME/GitHubDAT5"
    exit 0
fi
