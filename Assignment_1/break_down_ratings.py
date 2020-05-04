# Import Python-packages
from mrjob.job import MRJob
from mrjob.step import MRStep


class RatingsBreakDown (MRJob):
    MRJob.SORT_VALUES = True  # Set sort property of MRJob

    # Recepe-function
    def steps(self):
        return [  # 2 MRSteps in application
            MRStep(mapper=self.mapper_get_ratings,
                   reducer=self.reducer_count_ratings),
            MRStep(mapper=self.mapper_make_amounts_key,
                   reducer=self.reducer_output_results_for_single_reducer)
        ]

    # MRStep 1: Mapper
    def mapper_get_ratings(self, _, line):
        # Process the input (u.data) by creating a tuple and splitting on a TAB
        (userID, movieID, rating, timestamp) = line.split('\t')
        # Yield a key-value-pair
        yield int(movieID), int(rating)
        # The mapper automatically groups key-value-pairs
        # Therefore: 1:1, 2:4, 1:3, 2:2 etc. becomes 1:1,3 and 2:4,2 etc.

    # MRStep 1: Reducer
    def reducer_count_ratings(self, key, values):
        # Yield the key alongside the sum of the values belonging to the key
        yield key, sum(values)
        # Output example: 1:4, 2:6 etc.

    # MRStep 2: Mapper
    def mapper_make_amounts_key(self, movieID, ratingTotal):
        # Yield key-value-pair but format the ratingTotal so that the application sorts ascending correctly
        # If the ratingTotal is not formatted the output will be 1, 10, 11, 12, 2, 20, 21 etc.
        # Because of the formatting the output will be sorted correctly: 1, 2, 3, .... 10, 11, 12 etc.
        yield ('%020d' % int(ratingTotal), int(movieID))

    # MRStep 2: Reducer
    def reducer_output_results_for_single_reducer(self, ratingTotal, movieID):
        # Open the item file to include the movie database by creating tuples splitting on |
        # I've tried downloading the file from a server, but I keep running into permission errors
        # Therefore, opening the file has to be done with a static file-path
        with open('C:/Users/ldewa/Downloads/Hadoop/Assignment_1/u.item', 'r', encoding="ISO-8859-1") as m:
            movies = [tuple(map(str, l.split('|'))) for l in m]

        # For each key in keys...
        for id in movieID:
            # For each movie in movies
            for movie in movies:
                # Yield movie title and ratingTotal if the key and movieID match
                if str(movie[0]) == str(id):
                    yield str(movie[1]), int(ratingTotal)


# If the application is invoked, run RatingsBreakDown
if __name__ == '__main__':
    RatingsBreakDown.run()
