# LLRQ Documentation Site

This directory contains the GitHub Pages documentation site for the LLRQ (Log-Linear Reaction Quotient) package.

## Site Structure

- **`index.md`**: Main landing page with overview and interactive demo
- **`getting-started.md`**: Installation guide and first steps
- **`api-reference.md`**: Complete API documentation
- **`tutorials.md`**: Step-by-step tutorials and guides
- **`theory.md`**: Mathematical foundations and theory
- **`examples.md`**: Complete working examples
- **`_config.yml`**: Jekyll configuration
- **`_includes/`**: Reusable components (interactive demo, custom head)

## Local Development

To build the site locally:

```bash
# Install dependencies
bundle install

# Serve locally
bundle exec jekyll serve

# Visit http://localhost:4000
```

## Features

### Interactive Demo
- Real-time parameter adjustment
- Live visualization of LLRQ dynamics
- Educational tool for understanding the framework

### Mathematical Rendering
- MathJax for equations
- Proper LaTeX formatting
- Interactive mathematical content

### Code Examples
- Syntax highlighting
- Complete working examples
- Copy-paste ready code

### Responsive Design
- Mobile-friendly layout
- Clean, professional appearance
- Easy navigation

## Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the `gh-pages` branch.

## Content Organization

### For Beginners
1. **Getting Started**: Installation and basic usage
2. **Interactive Demo**: Hands-on exploration
3. **Simple Tutorial**: First reaction network

### For Users
1. **API Reference**: Complete documentation
2. **Tutorials**: Step-by-step guides
3. **Examples**: Working code samples

### For Researchers
1. **Theory**: Mathematical foundations
2. **Advanced Examples**: Complex networks
3. **Control Theory**: LQR applications

## Contributing

To add content:

1. **New pages**: Create `.md` files with proper front matter
2. **Update navigation**: Edit `_config.yml` header_pages
3. **Add examples**: Include in `examples.md` or create new files
4. **Interactive content**: Add to `_includes/` directory

## Technical Notes

- **Jekyll 4.3** with **Minima theme**
- **MathJax 3** for equations
- **Highlight.js** for syntax highlighting
- **Responsive CSS** for mobile compatibility
- **GitHub Pages** compatible

## File Formats

- **Markdown**: All content pages
- **HTML**: Interactive components
- **CSS**: Custom styling
- **JavaScript**: Interactive demos and functionality