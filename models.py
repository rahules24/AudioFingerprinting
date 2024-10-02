from peewee import Model, CharField, ForeignKeyField, SqliteDatabase

# Define your database connection
db = SqliteDatabase('fingerprint_database.db')

# Define your models
class Song(Model):
    title = CharField(unique=True)  # Assuming each song has a unique title

    class Meta:
        database = db

    def save_song(self):
        self.save()

    def add_fingerprint(self, fingerprint_data):
        Fingerprints.create(fingerprint_data=fingerprint_data, song=self)

    def add_fingerprints(self, fingerprint_data_list, anchortime_list):
        for fingerprint, anchortime in zip(fingerprint_data_list, anchortime_list):
            Fingerprints.create(fingerprint = fingerprint, anchortime = anchortime, song=self)

    def get_or_create(title):
        # Check if the song exists in the database
        existing_song = Song.select().where(Song.title == title).first()
        if existing_song:
            return existing_song
        else:
            # Create a new song entry in the database
            new_song = Song.create(title=title)
            return new_song

class Fingerprints(Model):
    song = ForeignKeyField(Song, backref='fingerprints')
    fingerprint = CharField()  # Assuming fingerprint is a string
    anchortime = CharField()  # Assuming fingerprint is a string

    class Meta:
        database = db

    def get_or_create(song, fingerprint):
        # Check if the fingerprint exists in the database
        existing_fingerprint = fingerprint.select().where((fingerprint.song == song) & (fingerprint.fingerprint == fingerprint)).first()
        if existing_fingerprint:
            return existing_fingerprint
        else:
            # Create a new fingerprint entry in the database
            new_fingerprint = fingerprint.create(song=song, fingerprint=fingerprint)
            return new_fingerprint
        
# Create tables if they don't exist
db.connect()
db.create_tables([Song, Fingerprints])
