    def remove_duplicates_and_reinitialize(self): ## fix later mabye :) srry future milez
        seen_indices = set()
        new_population = []
        for worm in self.population:
            non_zero_indices = list((np.where(worm.weight_matrix != self.original_genome)))
            indices_tuple = tuple(map(tuple, non_zero_indices))
            if (indices_tuple in seen_indices) and len(non_zero_indices[0]) < 1000:
                new_population.append(self.give_random_worm())
            else:
                seen_indices.add(indices_tuple)
                new_population.append(worm)

        self.population = new_population