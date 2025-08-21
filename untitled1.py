# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 20:33:53 2025

@author: User
"""

import os
from pydub import AudioSegment

# Folder containing .waptt.opus files
input_folder = "C:/Users/User/OneDrive/Desktop/Masters/thesis/DATA/"  # Change if needed

for file in os.listdir(input_folder):
    if file.endswith(".waptt.opus"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(input_folder, file.replace(".opus", ".mp3"))

        # Load and convert
        audio = AudioSegment.from_file(input_path, format="opus")
        audio.export(output_path, format="mp3")

        print(f"Converted: {file} → {output_path}")
