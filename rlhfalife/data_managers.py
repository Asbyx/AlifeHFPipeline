import pandas as pd
from typing import List, Any, Optional, Tuple, Iterator
from pathlib import Path


class DatasetManager:
    """
    Manages the dataset of simulation parameters, outputs, and videos.
    
    This class handles the storage and retrieval of simulation data, including:
    - Parameters (stored in out/params/)
    - Outputs (stored in out/outputs/)
    - Videos (stored in out/videos/)
    
    It maintains a CSV file (dataset.csv) with columns:
    hash,param_path,output_path,video_path
    """
    
    def __init__(self, dataset_path: str, out_paths: dict, simulator=None):
        """
        Initialize the DatasetManager.
        
        Args:
            dataset_path: Path to the dataset CSV file
            out_paths: Dictionary containing paths to various output directories
            simulator: Optional simulator instance for loading data
        """
        self.dataset_path = dataset_path
        self.out_paths = out_paths
        self.simulator = simulator
        self.data_df = self._load_or_create_dataset()
        
    def _load_or_create_dataset(self) -> pd.DataFrame:
        """Load the dataset from CSV or create a new one if it doesn't exist."""
        if Path(self.dataset_path).exists():
            return pd.read_csv(self.dataset_path, dtype=str)
        else:
            # Create a new dataset with the required columns
            df = pd.DataFrame(columns=['hash', 'param_path', 'output_path', 'video_path'])
            df.to_csv(self.dataset_path, index=False)
            return df
    
    def save(self):
        """Save the dataset to the CSV file."""
        self.data_df.to_csv(self.dataset_path, index=False)
    
    def reset(self):
        """
        Reset the dataset by deleting all files and clearing the dataframe.
        """
        # Get all file paths
        all_param_files = []
        all_output_files = []
        all_video_files = []
        
        for _, row in self.data_df.iterrows():
            if Path(row['param_path']).exists():
                all_param_files.append(row['param_path'])
            if Path(row['output_path']).exists():
                all_output_files.append(row['output_path'])
            if Path(row['video_path']).exists():
                all_video_files.append(row['video_path'])
                
        # Delete all files
        for file_path in all_param_files + all_output_files + all_video_files:
            try:
                Path(file_path).unlink()
            except (OSError, FileNotFoundError):
                pass  # Ignore errors if file doesn't exist or can't be deleted
                
        # Reset the dataframe
        self.data_df = pd.DataFrame(columns=['hash', 'param_path', 'output_path', 'video_path'])
        
        # Save the empty dataframe
        self.save()
    
    def add_entry(self, hash_value: str, param_path: str, output_path: str, video_path: str):
        """
        Add a new entry to the dataset.
        
        Args:
            hash_value: Hash value of the parameters
            param_path: Path to the saved parameters
            output_path: Path to the saved outputs
            video_path: Path to the saved video
        """
        # Check if the hash already exists
        if hash_value in self.data_df['hash'].values:
            # Update the existing entry
            self.data_df.loc[self.data_df['hash'] == hash_value, 'param_path'] = param_path
            self.data_df.loc[self.data_df['hash'] == hash_value, 'output_path'] = output_path
            self.data_df.loc[self.data_df['hash'] == hash_value, 'video_path'] = video_path
        else:
            # Add a new entry
            new_entry = pd.DataFrame({
                'hash': [hash_value],
                'param_path': [param_path],
                'output_path': [output_path],
                'video_path': [video_path]
            })
            self.data_df = pd.concat([self.data_df, new_entry], ignore_index=True)
        
        # Save the updated dataset
        self.save()
    
    def get_param_paths(self, hash_values: List[str]) -> List[Optional[str]]:
        """Get the paths to the parameters for a list of hashes."""
        param_paths = []
        for hash_value in hash_values:
            if hash_value in self.data_df['hash'].values:
                param_paths.append(self.data_df.loc[self.data_df['hash'] == hash_value, 'param_path'].iloc[0])
            else:
                raise ValueError(f"Hash {hash_value} not found in dataset")
        return param_paths
    
    def get_output_paths(self, hash_values: List[str]) -> List[Optional[str]]:
        """Get the paths to the outputs for a list of hashes."""
        output_paths = []
        for hash_value in hash_values:
            if hash_value in self.data_df['hash'].values:
                output_paths.append(self.data_df.loc[self.data_df['hash'] == hash_value, 'output_path'].iloc[0])
            else:
                raise ValueError(f"Hash {hash_value} not found in dataset")
        return output_paths
    
    def get_video_paths(self, hash_values: List[str]) -> List[Optional[str]]:
        """Get the paths to the videos for a list of hashes."""
        video_paths = []
        for hash_value in hash_values:
            if hash_value in self.data_df['hash'].values:
                video_paths.append(self.data_df.loc[self.data_df['hash'] == hash_value, 'video_path'].iloc[0])
            else:
                raise ValueError(f"Hash {hash_value} not found in dataset")
        return video_paths
    
    def load_param(self, hash_value: str) -> Any:
        """
        Load the parameters for a given hash.
        
        Args:
            hash_value: Hash value of the parameters to load
            
        Returns:
            The loaded parameters or None if not found
        """
        if not self.simulator:
            raise ValueError("Simulator not provided to DatasetManager. Cannot load parameters.")
            
        param_path = self.get_param_paths([hash_value])[0]
        if param_path:
            return self.simulator.load_param(param_path)
        return None
    
    def load_params(self, hash_values: List[str]) -> List[Any]:
        """
        Load multiple parameters for given hashes.
        
        Args:
            hash_values: List of hash values to load
            
        Returns:
            List of loaded parameters
        """
        return [self.load_param(hash_value) for hash_value in hash_values if self.get_param_paths([hash_value])[0]]
    
    def load_output(self, hash_value: str) -> Any:
        """
        Load the output for a given hash.
        """
        return self.simulator.load_output(self.get_output_paths([hash_value])[0])

    def load_outputs(self, hash_values: List[str]) -> List[Any]:
        """
        Load multiple outputs for given hashes.
        """
        return [self.load_output(hash_value) for hash_value in hash_values if self.get_output_paths([hash_value])[0]]

    def get_all_hashes(self) -> List[str]:
        """Get all hash values in the dataset."""
        return self.data_df['hash'].tolist()
    
    def add_entries_from_simulation(self, hashs: List[str], params: List[Any], 
                                   outputs: List[Any], param_paths: List[str], 
                                   output_paths: List[str], video_paths: List[str]):
        """
        Add multiple entries from a simulation run.
        
        Args:
            hashs: List of hash values
            params: List of parameters
            outputs: List of outputs
            param_paths: List of paths to saved parameters
            output_paths: List of paths to saved outputs
            video_paths: List of paths to saved videos
        """
        for i, hash_value in enumerate(hashs):
            self.add_entry(
                hash_value=str(hash_value),
                param_path=param_paths[i],
                output_path=output_paths[i],
                video_path=video_paths[i]
            )

    def __len__(self) -> int:
        """
        Get the number of entries in the dataset.
        """
        return len(self.data_df)


class PairsManager:
    """
    Manages pairs of simulations for comparison.
    
    This class handles the storage and retrieval of pairs of simulations, including:
    - Unranked pairs (where winner is empty)
    - Ranked pairs (where winner is filled)
    
    It maintains a CSV file (pairs.csv) with columns:
    hash1,hash2,winner
    """
    
    def __init__(self, pairs_path: str):
        """
        Initialize the PairsManager.
        
        Args:
            pairs_path: Path to the pairs CSV file
            dataset_manager: Optional DatasetManager instance for loading data
        """
        self.pairs_path = pairs_path
        self._load_or_create_pairs()
        
    def _load_or_create_pairs(self) -> None:
        """Load the pairs from CSV or create a new one if it doesn't exist."""
        if Path(self.pairs_path).exists():
            self.pairs_df = pd.read_csv(self.pairs_path, dtype={'hash1': str, 'hash2': str, 'winner': float})
        else:
            # Create a new pairs file with the required columns
            self.pairs_df = pd.DataFrame(columns=['hash1', 'hash2', 'winner'])
        self.save()

    def reset(self):
        """
        Reset the pairs manager by clearing all pairs and saving an empty pairs file.
        """
        self.pairs_df = pd.DataFrame(columns=['hash1', 'hash2', 'winner'])
        self.save()
        
    def reset_rankings(self):
        """
        Reset only the rankings by clearing all winners but keeping the pairs.
        This preserves the pairs structure but marks them all as unranked.
        """
        self.pairs_df['winner'] = None
        self.save()

    def _reindex_pairs(self):
        """
        Reorders the unranked pairs such that no hash appears n+1 times until all other hashes
        have appeared n times. This ensures users encounter a diverse range of simulations.
        """
        unranked_pairs = self.pairs_df[self.pairs_df['winner'].isna()].copy()

        # Create a copy to avoid modifying the original
        pairs = unranked_pairs.copy().reset_index(drop=True)
        
        # Initialize the count dictionary for all unique hashes
        all_hashes = pd.concat([pairs['hash1'], pairs['hash2']]).unique()
        hash_counts = {hash_val: 0 for hash_val in all_hashes}
        
        # Initialize the result dataframe
        ordered_pairs = pd.DataFrame(columns=pairs.columns)
        
        # Continue until all pairs are used
        remaining_pairs = pairs.copy()
        
        ordered_rows = []
        while not remaining_pairs.empty:
            scores_h1 = remaining_pairs['hash1'].map(hash_counts)
            scores_h2 = remaining_pairs['hash2'].map(hash_counts)
            remaining_pairs['_score'] = scores_h1 + scores_h2

            min_score_val = remaining_pairs['_score'].min()
            
            # Filter to pairs that have the minimum score
            min_score_pairs_df = remaining_pairs[remaining_pairs['_score'] == min_score_val]
            chosen_pair_series = min_score_pairs_df.sample(n=1).iloc[0]
            ordered_rows.append(chosen_pair_series[pairs.columns]) 
            
            # Update the hash counts for the hashes in the chosen pair
            hash_counts[chosen_pair_series['hash1']] += 1 
            hash_counts[chosen_pair_series['hash2']] += 1
            remaining_pairs = remaining_pairs.drop(chosen_pair_series.name)
        
        # After the loop, create the ordered_pairs DataFrame from the list of rows
        if ordered_rows:
            # Construct DataFrame from list of Series; ensure correct columns and order
            ordered_pairs = pd.DataFrame(ordered_rows, columns=pairs.columns).reset_index(drop=True)
        else:
            # If no unranked pairs, create an empty DataFrame with the correct columns
            ordered_pairs = pd.DataFrame(columns=pairs.columns)
        
        self.pairs_df = pd.concat([pd.DataFrame(ordered_pairs), self.pairs_df[self.pairs_df['winner'].notna()]], ignore_index=True).reset_index(drop=True)
        self.save()

    def save(self):
        """Save the pairs to the CSV file."""
        self.pairs_df.to_csv(self.pairs_path, index=False)
    
    def add_pairs(self, pairs: List[Tuple[str, str]], winners: Optional[List[float]] = None):
        """
        Add multiple pairs to the dataset. Optimized for faster execution.
        
        Args:
            pairs: List of (hash1, hash2) tuples
            winners: Optional list of winner values (0 for hash1, 1 for hash2, 0.5 for draw)
        """
        if not pairs:
            self._reindex_pairs()
            return

        effective_winners = [None] * len(pairs) if winners is None else winners
        if len(pairs) != len(effective_winners):
            raise ValueError("Length of pairs and winners must match.")

        input_df = pd.DataFrame(pairs, columns=['p_hash1', 'p_hash2'])
        input_df['p_winner'] = effective_winners
        input_df['p_id'] = input_df.index # Unique ID for each input pair

        # Use a temporary column for original index of self.pairs_df.
        # This helps in updating the correct rows in self.pairs_df.
        temp_df_indices_col = '__original_df_idx__'
        df_with_orig_idx = self.pairs_df.reset_index().rename(columns={'index': temp_df_indices_col})

        # Update existing pairs that match (hash1, hash2) order
        merged_fwd = pd.merge(
            df_with_orig_idx, input_df, 
            left_on=['hash1', 'hash2'], 
            right_on=['p_hash1', 'p_hash2'], 
            how='inner'
        )

        if not merged_fwd.empty:
            update_map_fwd = pd.Series(merged_fwd['p_winner'].values, index=merged_fwd[temp_df_indices_col])
            self.pairs_df.update(pd.Series(update_map_fwd, name='winner'))
        
        processed_p_ids = set(merged_fwd['p_id'].unique())

        # Update existing pairs that match (hash2, hash1) order
        remaining_input_df = input_df[~input_df['p_id'].isin(processed_p_ids)]
        if not remaining_input_df.empty:
            merged_bwd = pd.merge(
                df_with_orig_idx, remaining_input_df,
                left_on=['hash1', 'hash2'],
                right_on=['p_hash2', 'p_hash1'], # Note swapped columns
                how='inner'
            )
            if not merged_bwd.empty:
                bwd_adjusted_winners = merged_bwd['p_winner'].apply(
                    lambda w: (1.0 - w) if isinstance(w, (float, int)) else None
                )
                update_map_bwd = pd.Series(bwd_adjusted_winners.values, index=merged_bwd[temp_df_indices_col])
                self.pairs_df.update(pd.Series(update_map_bwd, name='winner'))
                processed_p_ids.update(merged_bwd['p_id'].unique())

        # Add new pairs
        new_pairs_to_add_df = input_df[~input_df['p_id'].isin(processed_p_ids)]
        if not new_pairs_to_add_df.empty:
            # Prepare for canonicalization: select relevant columns and copy
            temp_new_pairs = new_pairs_to_add_df[['p_hash1', 'p_hash2', 'p_winner']].copy()

            # Apply canonicalization: ensure p_hash1 <= p_hash2
            # Create mask for rows where p_hash1 > p_hash2
            mask_swap = temp_new_pairs['p_hash1'] > temp_new_pairs['p_hash2']
            
            # Store original p_hash1 for swapped rows to correctly assign to p_hash2
            original_p_hash1_for_swapped = temp_new_pairs.loc[mask_swap, 'p_hash1']
            
            # Perform swap for p_hash1 and p_hash2 where needed
            temp_new_pairs.loc[mask_swap, 'p_hash1'] = temp_new_pairs.loc[mask_swap, 'p_hash2']
            temp_new_pairs.loc[mask_swap, 'p_hash2'] = original_p_hash1_for_swapped
            
            # Adjust winner for swapped rows
            temp_new_pairs.loc[mask_swap, 'p_winner'] = temp_new_pairs.loc[mask_swap, 'p_winner'].apply(
                lambda w: (1.0 - w) if isinstance(w, (float, int)) else None
            )
            
            # Rename columns to final schema ('hash1', 'hash2', 'winner')
            temp_new_pairs.rename(columns={'p_hash1': 'hash1', 'p_hash2': 'hash2', 'p_winner': 'winner'}, inplace=True)
            
            # Deduplicate based on canonical (hash1, hash2), keeping the last occurrence
            # from the input batch (implicit by current DataFrame row order which preserves input order).
            final_new_rows = temp_new_pairs.drop_duplicates(subset=['hash1', 'hash2'], keep='last')

            # deduplicate based on hash1 and hash2 such that hash1 != hash2
            final_new_rows = final_new_rows[final_new_rows['hash1'] != final_new_rows['hash2']]
            
            # Concatenate the unique, canonical new pairs, if any remain after deduplication
            if not final_new_rows.empty:
                self.pairs_df = pd.concat([self.pairs_df, final_new_rows], ignore_index=True)
        
        self._reindex_pairs() # This method calls self.save()
    
    def _get_unranked_pairs(self) -> pd.DataFrame:
        """Get all unranked pairs (where winner is null)."""
        return self.pairs_df[self.pairs_df['winner'].isnull()].reset_index(drop=True).copy()
    
    def _get_ranked_pairs(self) -> pd.DataFrame:
        """Get all ranked pairs (where winner is not null)."""
        return self.pairs_df[self.pairs_df['winner'].notnull()].reset_index(drop=True).copy()
    
    def unranked_pairs_iterator(self) -> Iterator[Tuple[str, str]]:
        """Get an iterator over all unranked pairs."""
        return self._get_unranked_pairs()[['hash1', 'hash2']].itertuples(index=False, name=None)

    def set_winner(self, hash1: str, hash2: str, winner: float):
        """
        Set the winner for a pair.
        
        Args:
            hash1: Hash value of the first simulation
            hash2: Hash value of the second simulation
            winner: float in [0, 1] (0=hash1 wins, 1=hash2 wins, 0.5=draw)
        """
        if winner < 0 or winner > 1:
            raise ValueError(f"Winner ({winner}) must be a float in [0, 1]")
        
        # Find the pair
        pair_idx = self.pairs_df[
            ((self.pairs_df['hash1'] == hash1) & (self.pairs_df['hash2'] == hash2)) |
            ((self.pairs_df['hash1'] == hash2) & (self.pairs_df['hash2'] == hash1))
        ].index
        
        if len(pair_idx) > 0:
            self.pairs_df.at[pair_idx[0], 'winner'] = winner if self.pairs_df.at[pair_idx[0], 'hash1'] == hash1 else 1 - winner
        else:
            self.add_pairs([(hash1, hash2)], [winner])
    
    def _get_all_pairs(self) -> pd.DataFrame:
        """Get all pairs."""
        return self.pairs_df.copy()

    def get_nb_unranked_pairs(self) -> int:
        """Get the number of unranked pairs."""
        return self._get_unranked_pairs().shape[0]
    
    def get_nb_ranked_pairs(self) -> int:
        """Get the number of ranked pairs."""
        return self._get_ranked_pairs().shape[0]
    
    def get_nb_pairs(self) -> int:
        """Get the total number of pairs."""
        return self.pairs_df.shape[0]

    def get_next_unranked_pair(self) -> Tuple[str, str]:
        """Get the next unranked pair."""
        unranked_pairs = self._get_unranked_pairs()
        if unranked_pairs.empty:
            return None, None
        return unranked_pairs.iloc[0]['hash1'], unranked_pairs.iloc[0]['hash2']
    
    def get_last_ranked_pair(self) -> Tuple[str, str]:
        """Get the last ranked pair."""
        ranked_pairs = self._get_ranked_pairs()
        if ranked_pairs.empty:
            return None, None
        return ranked_pairs.iloc[-1]['hash1'], ranked_pairs.iloc[-1]['hash2']

class TrainingDataset:
    """
    Manages the data for the training of the rewarder
    """
    def __init__(self, pairs_manager: PairsManager, dataset_manager: DatasetManager):
        """
        Initialize the TrainingDataset.
        This dataset delivers triplets of (path_to_output_1, path_to_output_2, winner). The pairs are all ranked. The load of the outputs is expected to be done in the Rewarder, from the paths given by the TrainingDataset.

        Args:
            pairs_manager: The pairs manager
            dataset_manager: The dataset manager
        """        
        ranked_pairs = pairs_manager._get_ranked_pairs()
        
        # Prepare the data
        self.data = []
        self.simulations = set()
        for _, row in ranked_pairs.iterrows():
            hash1 = row['hash1']
            hash2 = row['hash2']

            winner = float(row['winner'])

            self.simulations.add(hash1)
            self.simulations.add(hash2)
            
            # Get the output paths
            output_path1 = dataset_manager.get_output_paths([hash1])[0]
            output_path2 = dataset_manager.get_output_paths([hash2])[0]
            
            self.data.append((output_path1, output_path2, winner))
        self.simulations_number = len(self.simulations)

    def get_simulations_hashes(self) -> List[str]:
        """Get the hashes of the simulations in the dataset."""
        return list(self.simulations)

    def __len__(self) -> int:
        """
        Get the number of pairs in the dataset
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, str, float]:
        """
        Get the pair at the given index.

        Returns:
            A tuple containing the path_to_output_1, path_to_output_2, winner (float in [0, 1], 0 for hash1 wins, 1 for hash2 wins, 0.5 for draw)
        """
        if index < 0 or index >= len(self.data):
            raise IndexError("Index out of bounds")
        return self.data[index]

    def __iter__(self) -> Iterator[Tuple[str, str, float]]:
        """
        Get an iterator over the pairs

        Returns:
            An iterator over the pairs
        """
        return iter(self.data)
        
