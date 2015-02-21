require "rubygems"
require "active_record"

config = YAML.load_file("./blog_db.yml")
ActiveRecord::Base.establish_connection(config["db"]["development"])

class Post < ActiveRecord::Base
end


# all
p Post.all
puts ""

# first
p Post.first
puts ""

# last
p Post.last.title
puts ""

# find
p Post.find(2)
puts ""

# find_by
p Post.find_by_title("title_3");
puts ""

# find_by (mutli fields)
p Post.find_by_title_and_id("title_3", 2);




