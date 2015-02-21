require "rubygems"
require "active_record"

config = YAML.load_file("./blog_db.yml")
ActiveRecord::Base.establish_connection(config["db"]["development"])

class Post < ActiveRecord::Base
end
post = Post.new({:title => "title_1", :body => "body_1"});
post.save

p Post.all


