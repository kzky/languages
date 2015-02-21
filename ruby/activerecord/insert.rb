require "rubygems"
require "active_record"

config = YAML.load_file("./blog_db.yml")
ActiveRecord::Base.establish_connection(config["db"]["development"])

class Post < ActiveRecord::Base
end

# new then save
post = Post.new({:title => "title_2", :body => "body_2"});
post.save

# new and block then save
post = Post.new do |p|
  p.title ="title_3"
  p.body = "body_3"
end
post.save

# create
Post.create({:title => "title_4", :body => "body_4"})

p Post.all


