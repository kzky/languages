require "rubygems"
require "active_record"
require 'logger'

config = YAML.load_file("./blog_db.yml")
ActiveRecord::Base.establish_connection(config["db"]["development"])
ActiveRecord::Base.logger = Logger.new(STDOUT); ## for debugging

class Post < ActiveRecord::Base
  #scope :top3, order("created_at").limit(3) # deprecated
  scope :top3, -> {order("created_at").limit(3)}
end

# save
post = Post.find(1);
post.title = "(new) title_1"
post.save
p Post.find(1)

# update_attribute
post = Post.find(1);
post.update_attribute(:title, "(new2) title_1");
p Post.find(1)

post = Post.find(1);
post.update_attributes(:title => "(new2) title_1", :title => "(new) body_1");
p Post.find(1)

Post.where(:id => 1..2).update_all(:title => "nnn2", :body => "hhh2")
p Post.order("id asc").limit(3)
