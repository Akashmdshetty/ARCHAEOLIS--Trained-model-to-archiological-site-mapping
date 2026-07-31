import pandas as pd
import io
import os

class BoreholeLogParser:
    """
    Utility to parse LAS (Log Ascii Standard) and similar text logging data.
    """
    @staticmethod
    def parse_las(file_path):
        """Minimal LAS parser for extraction of Curve Data section."""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        data_start = -1
        columns = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('~Curve'):
                # Collect column names from curve info section
                j = i + 1
                while j < len(lines) and not lines[j].startswith('~'):
                    if lines[j].strip() and not lines[j].strip().startswith('#'):
                        col = lines[j].split('.')[0].strip()
                        if col:
                            columns.append(col)
                    j += 1
            elif line.startswith('~A'):
                data_start = i + 1
                break
        
        if data_start == -1:
            return None
            
        data_text = "".join(lines[data_start:])
        df = pd.read_csv(io.StringIO(data_text), sep=r'\s+', names=columns)
        return df

    @staticmethod
    def parse_txt_log(file_path):
        """Parses tabulated text logs like the ones found in 'Standard Data'."""
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Find header line (usually contains DEPTH_WSF or similar)
        header_idx = -1
        for i, line in enumerate(lines):
            if 'DEPTH' in line or 'm' in line:
                header_idx = i
                break
        
        if header_idx == -1:
            # Fallback to simple read
            return pd.read_csv(file_path, sep=r'\s+')
            
        # The data usually starts 2 lines after the header start if there are units
        data_start = header_idx + 2
        cols = lines[header_idx].split()
        
        data_text = "".join(lines[data_start:])
        df = pd.read_csv(io.StringIO(data_text), sep=r'\s+', names=cols)
        return df

def get_borehole_data(root_dir):
    """Scan directory for known log files and return them as a dict of DataFrames."""
    results = {}
    
    # Target files from the provided dataset
    targets = [
        ('LAS', 'logging_data/347-M0065A_logging_data/Log Ascii Standard (LAS) Data/Processed Data/347-M0065A_mai_mcg.las'),
        ('MAI', 'logging_data/347-M0065A_logging_data/Standard Data/347-M0065A_mai.txt')
    ]
    
    parser = BoreholeLogParser()
    
    for label, rel_path in targets:
        full_path = os.path.join(root_dir, rel_path)
        if os.path.exists(full_path):
            try:
                if rel_path.endswith('.las'):
                    results[label] = parser.parse_las(full_path)
                else:
                    results[label] = parser.parse_txt_log(full_path)
            except Exception as e:
                print(f"Error parsing {label}: {e}")
                
    return results
