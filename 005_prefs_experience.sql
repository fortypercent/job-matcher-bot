ALTER TABLE user_preferences
    ADD COLUMN IF NOT EXISTS show_without_salary BOOLEAN DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS experience TEXT[] DEFAULT '{"noExperience","between1And3","between3And6","moreThan6"}';
