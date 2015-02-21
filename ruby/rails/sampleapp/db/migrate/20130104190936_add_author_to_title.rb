class AddAuthorToTitle < ActiveRecord::Migration
  def change
    add_column :tests, :author, :string
  end
end
