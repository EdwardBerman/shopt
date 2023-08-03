#!/bin/bash

yaml_file="shopt.yml"
new_comment="$1"

sed -i "s/^\s*CommentsOnRun:.*/  CommentsOnRun: \"$new_comment\"/" "$yaml_file"
echo "Comments on the run have been updated in the YAML file."


